"""GAT 输出、注意力系数、梯度与边界验收。"""
import unittest
import torch
import dgl
from dgl.nn import GATConv
from gat import GAT, neighbor_softmax, add_self_loops


class GATTest(unittest.TestCase):
    comparisons = 0
    max_output_error = 0.0
    max_alpha_error = 0.0
    max_grad_error = 0.0

    def test_dgl_parity(self):
        graphs = {
            "directed_isolated": (5, [[0,0,1,3],[1,2,2,0]]),
            "loops_duplicates": (4, [[0,0,0,1,2,2],[0,1,1,2,2,2]]),
            "no_edges": (3, [[],[]]),
            "no_nodes": (0, [[],[]]),
            "one_node": (1, [[0,0],[0,0]]),
        }
        devices = [("cpu",torch.float64)]
        if torch.cuda.is_available():
            devices.append(("cuda",torch.float32))
        for device,dtype in devices:
            for name,(n,pairs) in graphs.items():
                for loops in [False,True]:
                    for heads in [1,4]:
                        for concat in [False,True]:
                            with self.subTest(device=device,graph=name,loops=loops,
                                              heads=heads,concat=concat):
                                edges=torch.tensor(pairs,dtype=torch.long,device=device)
                                graph=dgl.graph((edges[0],edges[1]),num_nodes=n,device=device)
                                if loops:
                                    edges=add_self_loops(edges,n)
                                    graph=dgl.add_self_loop(dgl.remove_self_loop(graph))
                                manual=GAT(3,2,heads,concat).to(device=device,dtype=dtype)
                                reference=GATConv(3,2,heads,allow_zero_in_degree=True).to(device=device,dtype=dtype)
                                with torch.no_grad():
                                    manual.bias.uniform_(-0.2,0.2)
                                    reference.load_state_dict(manual.state_dict())
                                a=torch.randn(n,3,device=device,dtype=dtype,requires_grad=True)
                                b=a.detach().clone().requires_grad_()
                                out,alpha=manual(a,edges,True)
                                expected,weights=reference(graph,b,get_attention=True)
                                expected=expected.flatten(1) if concat else expected.mean(1)
                                atol,rtol=(1e-10,1e-9) if dtype==torch.float64 else (3e-5,3e-4)
                                torch.testing.assert_close(out,expected,atol=atol,rtol=rtol)
                                torch.testing.assert_close(alpha,weights.squeeze(-1),atol=atol,rtol=rtol)
                                sums=alpha.new_zeros((n,heads))
                                sums.index_add_(0,edges[1],alpha)
                                degree=torch.bincount(edges[1],minlength=n)
                                torch.testing.assert_close(sums[degree>0],torch.ones_like(sums[degree>0]),atol=atol,rtol=rtol)
                                upstream=torch.randn_like(out)
                                ga=torch.autograd.grad(out,(a,*manual.parameters()),upstream)
                                gb=torch.autograd.grad(expected,(b,*(reference.get_parameter(name) for name,_ in manual.named_parameters())),upstream)
                                for left,right in zip(ga,gb):
                                    torch.testing.assert_close(left,right,atol=atol,rtol=rtol)
                                    err=(left-right).abs().max().item() if left.numel() else 0
                                    type(self).max_grad_error=max(type(self).max_grad_error,err)
                                for actual,wanted,field in [(out,expected,"max_output_error"),(alpha,weights.squeeze(-1),"max_alpha_error")]:
                                    err=(actual-wanted).abs().max().item() if actual.numel() else 0
                                    setattr(type(self),field,max(getattr(type(self),field),err))
                                type(self).comparisons+=1

    def test_softmax(self):
        scores=torch.tensor([[1000.,-1000.],[1001.,-1001.],[-9999.,9999.]],dtype=torch.float64,requires_grad=True)
        dst=torch.tensor([1,1,2])
        result=neighbor_softmax(scores,dst,4)
        expected=torch.cat((scores[:2].softmax(0),torch.ones_like(scores[2:])))
        torch.testing.assert_close(result,expected)
        offsets=torch.tensor([[500.,100.],[500.,100.],[-500.,-100.]])
        torch.testing.assert_close(neighbor_softmax(scores+offsets,dst,4),result)
        self.assertTrue(torch.autograd.gradcheck(lambda x:neighbor_softmax(x,dst,4),(scores,)))

    def test_uniform_ablation(self):
        layer=GAT(1,1,heads=2,concat=True,attention=False).double()
        with torch.no_grad():
            layer.fc.weight.fill_(1)
        x=torch.tensor([[2.],[8.],[4.]],dtype=torch.float64)
        edges=torch.tensor([[0,1,1],[2,2,2]])
        out,alpha=layer(x,edges,True)
        torch.testing.assert_close(alpha,torch.full((3,2),1/3,dtype=torch.float64))
        torch.testing.assert_close(out,torch.tensor([[0.,0.],[0.,0.],[6.,6.]],dtype=torch.float64))
        self.assertFalse(any("attn" in name for name,_ in layer.named_parameters()))

    def test_aggregation_and_permutation(self):
        x=torch.randn(5,3,dtype=torch.float64)
        edges=add_self_loops(torch.tensor([[0,1,2,2],[1,2,0,3]]),5)
        concat=GAT(3,2,heads=4).double()
        mean=GAT(3,2,heads=4,concat=False).double()
        mean.load_state_dict(concat.state_dict())
        torch.testing.assert_close(mean(x,edges),concat(x,edges).reshape(5,4,2).mean(1))
        order=torch.randperm(edges.shape[1])
        torch.testing.assert_close(concat(x,edges[:,order]),concat(x,edges))
        looped=add_self_loops(torch.tensor([[0,0,0],[0,0,1]]),3)
        self.assertEqual((looped[0]==looped[1]).sum().item(),3)

    def test_invalid_inputs(self):
        layer=GAT(3,2)
        x=torch.randn(4,3)
        for edges in [torch.zeros(2,1),torch.zeros(3,1,dtype=torch.long),
                      torch.tensor([[-1],[0]]),torch.tensor([[0],[4]])]:
            with self.subTest(edges=edges):
                with self.assertRaises(ValueError): layer(x,edges)
        for values in [torch.randn(4,2),torch.full((4,3),float("nan")),
                       torch.full((4,3),float("inf"))]:
            with self.subTest(shape=values.shape):
                with self.assertRaises(ValueError): layer(values,torch.empty(2,0,dtype=torch.long))

    @classmethod
    def tearDownClass(cls):
        print(f"PARITY comparisons={cls.comparisons} output={cls.max_output_error:.9e} "
              f"alpha={cls.max_alpha_error:.9e} gradient={cls.max_grad_error:.9e}",flush=True)


if __name__=="__main__":
    torch.set_num_threads(1)
    torch.manual_seed(42)
    print(f"torch={torch.__version__} dgl={dgl.__version__} cuda={torch.cuda.is_available()}",flush=True)
    unittest.main(verbosity=2)
