
#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_NonLocalBC.H>

using namespace amrex;

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        Box domain(IntVect(0), IntVect(63,47,63));
        BoxArray ba(domain);
        ba.maxSize(IntVect(32,16,32));
        DistributionMapping dm(ba);

        iMultiFab mf(ba, dm, 3, 2);
        for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
            Box const& bx = mfi.fabbox();
            Array4<int> const& fab = mf.array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                fab(i,j,k,0) = i;
                if (j < 0) {
                    fab(i,j,k,1) = j + domain.length(1);
                } else if (j > domain.bigEnd(1)) {
                    fab(i,j,k,1) = j - domain.length(1);
                } else {
                    fab(i,j,k,1) = j;
                }
                fab(i,j,k,2) = k;
            });
        }

        NonLocalBC::FillPolar(mf, 0, mf.nComp(), mf.nGrowVect(), domain);

        Box vdomain = domain;
#if (AMREX_SPACEDIM == 3)
        vdomain.grow(2,2);
#endif
        const int jmid = domain.length(1)/2;
        const int ilen = domain.length(0);
        const int jlen = domain.length(1);
#if (AMREX_SPACEDIM == 3)
        const int klen = domain.length(2);
#endif

        for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
            Box const& bx = mfi.fabbox();
            Array4<int const> const& fab = mf.array(mfi);
            amrex::LoopOnCpu(bx, [=] (int i, int j, int k) noexcept
            {
                int ii = fab(i,j,k,0);
                int jj = fab(i,j,k,1);
                int kk = fab(i,j,k,2);
                IntVect iv{IntVect(AMREX_D_DECL(i,j,k))};
                IntVect iv2{IntVect(AMREX_D_DECL(ii,jj,kk))};
                if (kk != k) {
                    amrex::AllPrint() << iv << " " << iv2 << std::endl;
                    amrex::Abort("kk != k");
                }
                if (vdomain.contains(iv)) {
                    if (ii != i or jj != j) {
                        amrex::AllPrint() << iv << " " << iv2 << std::endl;
                        amrex::Abort("vdomain");
                    }
                } else if (amrex::adjCellLo(vdomain, 0, 2).contains(iv)) {
                    int jtmp = j + jmid;
                    if (jtmp >= jlen) {
                        jtmp -= jlen;
                    }
                    if ((ii+i != -1) or jtmp != jj) {
                        amrex::AllPrint() << iv << " " << iv2 << std::endl;
                        amrex::Abort("lo-x");
                    }
                } else if (amrex::adjCellHi(vdomain, 0, 2).contains(iv)) {
                    int jtmp = j + jmid;
                    if (jtmp >= jlen) {
                        jtmp -= jlen;
                    }
                    if ((ii+i != 2*ilen-1) or jtmp != jj) {
                        amrex::AllPrint() << iv << " " << iv2 << std::endl;
                        amrex::Abort("hi-x");
                    }
                } else if (amrex::adjCellLo(vdomain, 1, 2).contains(iv)) {
                    if (ii != i or jj != j+jlen) {
                        amrex::AllPrint() << iv << " " << iv2 << std::endl;
                        amrex::Abort("lo-y");
                    }
                } else if (amrex::adjCellHi(vdomain, 1, 2).contains(iv)) {
                    if (ii != i or jj != j-jlen) {
                        amrex::AllPrint() << iv << " " << iv2 << std::endl;
                        amrex::Abort("hi-y");
                    }
                } else if (Box(IntVect(AMREX_D_DECL(-2,-2,-2)),
                               IntVect(AMREX_D_DECL(-1,-1,klen+1))).contains(iv)) {
                    if (ii+i != -1 or jj != j+jmid) {
                        amrex::AllPrint() << iv << " " << iv2 << std::endl;
                        amrex::Abort("lo-x/lo-y");
                    }
                } else if (Box(IntVect(AMREX_D_DECL(-2,jlen,-2)),
                               IntVect(AMREX_D_DECL(-1,jlen+1,klen+1))).contains(iv)) {
                    if (ii+i != -1 or jj != j-jmid) {
                        amrex::AllPrint() << iv << " " << iv2 << std::endl;
                        amrex::Abort("lo-x/hi-y");
                    }
                } else if (Box(IntVect(AMREX_D_DECL(ilen,-2,-2)),
                               IntVect(AMREX_D_DECL(ilen+1,-1,klen+1))).contains(iv)) {
                    if (ii+i != 2*ilen-1 or jj != j+jmid) {
                        amrex::AllPrint() << iv << " " << iv2 << std::endl;
                        amrex::Abort("hi-x/lo-y");
                    }
                } else if (Box(IntVect(AMREX_D_DECL(ilen,jlen,-2)),
                               IntVect(AMREX_D_DECL(ilen+1,jlen+1,klen+1))).contains(iv)) {
                    if (ii+i != 2*ilen-1 or jj != j-jmid) {
                        amrex::AllPrint() << iv << " " << iv2 << std::endl;
                        amrex::Abort("hi-x/hi-y");
                    }
                } else {
                    amrex::AllPrint() << "iv = " << iv << std::endl;
                    amrex::Abort("should not get here");
                }
            });
        }
    }
    amrex::Finalize();
}

