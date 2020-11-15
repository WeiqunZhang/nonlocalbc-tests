
#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_NonLocalBC.H>

using namespace amrex;

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        Box domain(IntVect(0), IntVect(63));
        BoxArray ba(domain);
        ba.maxSize(IntVect(AMREX_D_DECL(32,16,32)));
        DistributionMapping dm(ba);

        iMultiFab mf(ba, dm, 3, 2);
        for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
            Box const& bx = mfi.fabbox();
            Array4<int> const& fab = mf.array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                fab(i,j,k,0) = i;
                fab(i,j,k,1) = j;
                fab(i,j,k,2) = k;
            });
        }

        NonLocalBC::Rotate90(mf, 0, mf.nComp(), mf.nGrowVect(), domain);

        for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
            Box const& bx = mfi.fabbox();
            Array4<int const> const& fab = mf.array(mfi);
            amrex::LoopOnCpu(bx, [=] (int i, int j, int k) noexcept
            {
                int ii = fab(i,j,k,0);
                int jj = fab(i,j,k,1);
                int kk = fab(i,j,k,2);
                if (kk != k) {
                    amrex::AllPrint() << IntVect(AMREX_D_DECL(i,j,k)) << " "
                                      << IntVect(AMREX_D_DECL(ii,jj,kk)) << std::endl;
                    amrex::Abort("kk != k");
                }
                if (i >= 0 and j >= 0) {
                    if (ii != i or jj != j) {
                        amrex::AllPrint() << IntVect(AMREX_D_DECL(i,j,k)) << " "
                                          << IntVect(AMREX_D_DECL(ii,jj,kk)) << std::endl;
                        amrex::Abort("Quandrant I");
                    }
                }
                if (i < 0 and j >= 0) {
                    if (ii != j or jj != (-i-1)) {
                        amrex::AllPrint() << IntVect(AMREX_D_DECL(i,j,k)) << " "
                                          << IntVect(AMREX_D_DECL(ii,jj,kk)) << std::endl;
                        amrex::Abort("Quandrant II");
                    }
                }
                if (i < 0 and j < 0) {
                    if ((ii+i) != -1 or (jj+j) != -1) {
                        amrex::AllPrint() << IntVect(AMREX_D_DECL(i,j,k)) << " "
                                          << IntVect(AMREX_D_DECL(ii,jj,kk)) << std::endl;
                        amrex::Abort("Quandrant III");
                    }
                }
                if (i >= 0 and j < 0) {
                    if (ii != (-j-1) or jj != i) {
                        amrex::AllPrint() << IntVect(AMREX_D_DECL(i,j,k)) << " "
                                          << IntVect(AMREX_D_DECL(ii,jj,kk)) << std::endl;
                        amrex::Abort("Quandrant IV");
                    }
                }
            });
        }
    }
    amrex::Finalize();
}

