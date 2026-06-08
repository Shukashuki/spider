"""LQR Benchmark v2: Open-loop convergence curves.

Nominal + Process Noise, all open-loop 128 iters.
Pure MPPI vs Annealing MPPI vs MPPI+CMA (σ₀=0.5).
"""

from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.linalg import solve_discrete_are


@dataclass
class LQRProblem:
    nx: int = 4; nu: int = 2; horizon: int = 20; dt: float = 0.1
    process_noise_std: float = 0.0

    def __post_init__(self):
        self.A = np.eye(self.nx); self.A[0,2]=self.dt; self.A[1,3]=self.dt
        self.B = np.zeros((self.nx,self.nu)); self.B[2,0]=self.dt; self.B[3,1]=self.dt
        self.Q = np.diag([10.,10.,1.,1.]); self.R = np.eye(self.nu)*0.1
        self.P = solve_discrete_are(self.A, self.B, self.Q, self.R)
        self.K = np.linalg.inv(self.R + self.B.T@self.P@self.B) @ (self.B.T@self.P@self.A)
        self.x0 = np.array([1.0, -0.5, 0.2, -0.1])

    def optimal_cost(self):
        x=self.x0.copy(); cost=0.
        for _ in range(self.horizon):
            u=-self.K@x; cost+=x@self.Q@x+u@self.R@u; x=self.A@x+self.B@u
        return cost + x@self.Q@x

    def rollout_cost_batch(self, U_batch):
        dev=U_batch.device
        A=torch.tensor(self.A,dtype=torch.float32,device=dev)
        B=torch.tensor(self.B,dtype=torch.float32,device=dev)
        Q=torch.tensor(self.Q,dtype=torch.float32,device=dev)
        R=torch.tensor(self.R,dtype=torch.float32,device=dev)
        x0=torch.tensor(self.x0,dtype=torch.float32,device=dev)
        N=U_batch.shape[0]; x=x0.unsqueeze(0).expand(N,-1).clone()
        costs=torch.zeros(N,device=dev)
        for k in range(self.horizon):
            u=U_batch[:,k,:]; costs+=(x@Q*x).sum(1)+(u@R*u).sum(1)
            x=x@A.T+u@B.T
            if self.process_noise_std>0: x=x+self.process_noise_std*torch.randn_like(x)
        return costs+(x@Q*x).sum(1)


@dataclass
class Cfg:
    N:int=1024; iters:int=128; sigma0:float=0.5; temp:float=1.0; device:str="cpu"


def _w(costs, temp):
    j=costs.min(); w=torch.exp(-(costs-j)/temp); return w/w.sum()


def run_pure_mppi(prob, cfg, seed=0):
    torch.manual_seed(seed); d=prob.horizon*prob.nu
    mu=torch.zeros(d,device=cfg.device); hist=[]
    for _ in range(cfg.iters):
        eps=torch.randn(cfg.N,d,device=cfg.device)*cfg.sigma0
        U=mu.unsqueeze(0)+eps; U[0]=mu; eps[0]=0.
        c=prob.rollout_cost_batch(U.reshape(cfg.N,prob.horizon,prob.nu))
        hist.append(float(c[0])); w=_w(c,cfg.temp); mu=(w[:,None]*U).sum(0)
    return hist


def run_anneal_mppi(prob, cfg, seed=0, beta=0.9):
    torch.manual_seed(seed); d=prob.horizon*prob.nu
    mu=torch.zeros(d,device=cfg.device); hist=[]
    for it in range(cfg.iters):
        ns=cfg.sigma0*(beta**it)
        eps=torch.randn(cfg.N,d,device=cfg.device)*ns
        U=mu.unsqueeze(0)+eps; U[0]=mu; eps[0]=0.
        c=prob.rollout_cost_batch(U.reshape(cfg.N,prob.horizon,prob.nu))
        hist.append(float(c[0])); w=_w(c,cfg.temp); mu=(w[:,None]*U).sum(0)
    return hist


def run_mppi_cma(prob, cfg, seed=0, eta_mu=0.5, eta_sig=0.3):
    torch.manual_seed(seed); d=prob.horizon*prob.nu; jit=1e-4
    mu=torch.zeros(d,device=cfg.device)
    Sig=(cfg.sigma0**2)*torch.eye(d,device=cfg.device); hist=[]
    for _ in range(cfg.iters):
        S=0.5*(Sig+Sig.T)
        try: L=torch.linalg.cholesky(S+jit*torch.eye(d,device=cfg.device))
        except: L=torch.diag(S.diag().clamp(min=jit).sqrt())
        z=torch.randn(cfg.N,d,device=cfg.device); eps=z@L.T
        U=mu.unsqueeze(0)+eps; U[0]=mu; eps[0]=0.
        c=prob.rollout_cost_batch(U.reshape(cfg.N,prob.horizon,prob.nu))
        hist.append(float(c[0])); w=_w(c,cfg.temp)
        wm=(w[:,None]*eps).sum(0); mu_new=mu+wm
        mu=(1-eta_mu)*mu+eta_mu*mu_new
        sw=w.sqrt(); we=sw[:,None]*eps; Ss=we.T@we
        Sig=(1-eta_sig)*Sig+eta_sig*Ss+jit*torch.eye(d,device=cfg.device)
        Sig=0.5*(Sig+Sig.T)
    return hist


def main():
    dev="cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {dev}")
    cfg=Cfg(N=1024,iters=128,sigma0=0.5,temp=1.0,device=dev)
    seeds=5

    scenarios={
        "Nominal (No Noise)": LQRProblem(process_noise_std=0.0),
        "Process Noise (σ=0.05)": LQRProblem(process_noise_std=0.05),
    }
    opts={
        "Pure MPPI": lambda p,s: run_pure_mppi(p,cfg,seed=s),
        "Annealing MPPI (β=0.9)": lambda p,s: run_anneal_mppi(p,cfg,seed=s),
        "MPPI+CMA (η_μ=0.5)": lambda p,s: run_mppi_cma(p,cfg,seed=s,eta_mu=0.5),
    }
    clrs={"Pure MPPI":"#1f77b4","Annealing MPPI (β=0.9)":"#ff7f0e","MPPI+CMA (η_μ=0.5)":"#d62728"}

    fig,axes=plt.subplots(1,2,figsize=(14,5.5))
    iters=np.arange(1,cfg.iters+1)

    for idx,(sc_name,prob) in enumerate(scenarios.items()):
        ax=axes[idx]; opt_cost=prob.optimal_cost()
        print(f"\n{'='*60}\n{sc_name}\n{'='*60}")
        print(f"  LQR Optimal: {opt_cost:.4f}")

        for oname,fn in opts.items():
            runs=[fn(prob,s) for s in range(seeds)]
            data=np.array(runs)
            m=data.mean(0); s=data.std(0)
            fm,fs=data[:,-1].mean(),data[:,-1].std()
            gap=(fm-opt_cost)/abs(opt_cost)*100
            print(f"  {oname:30s}: {fm:.4f} ± {fs:.4f}  (gap={gap:+.2f}%)")
            ax.plot(iters,m,label=oname,color=clrs[oname],linewidth=2)
            ax.fill_between(iters,m-s,m+s,alpha=0.15,color=clrs[oname])

        ax.axhline(opt_cost,color="black",linestyle="--",linewidth=1.5,
                   label=f"LQR Optimal ({opt_cost:.1f})")
        ax.set_xlabel("Iteration"); ax.set_ylabel("Cost (exploit)")
        ax.set_title(sc_name); ax.legend(fontsize=8); ax.grid(True,alpha=0.3)

    plt.suptitle("LQR Open-Loop Convergence: 1024 samples × 128 iters × 5 seeds, σ₀=0.5",fontsize=12)
    plt.tight_layout()
    out="/home/roy/.openclaw/workspace/spider/outputs/lqr_benchmark_v2.png"
    plt.savefig(out,dpi=150,bbox_inches="tight"); print(f"\nSaved: {out}"); plt.close()


if __name__=="__main__":
    main()
