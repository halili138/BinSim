#include "cuda_hvec.cuh"
#include "cuda_expm.cuh"
#include "cuda_grad.cuh"
#include "cuda_backgrad.cuh"
#include "cuda_batchgrad.cuh"

extern "C"
{
    void sync_device_cuda()
    {
        cudaDeviceSynchronize();
    }

    void destroy_basisdev_f64(void *basis_ptr)
    {
        if (basis_ptr == nullptr)
            return;

        BasisViewDev<uint32> *basis = static_cast<BasisViewDev<uint32> *>(basis_ptr);

        basis->clear();
        delete basis;
    }

    void *build_basisdev_f64(void *host_basis_ptr)
    {
        const BasisManager<uint32> *h_basis = static_cast<BasisManager<uint32> *>(host_basis_ptr);

        return upload_basis<uint32>(h_basis);
    }

    void destroy_networkdev_f64(void *net_ptr)
    {
        if (net_ptr == nullptr)
            return;

        NetworkDev<uint32, double> *net = static_cast<NetworkDev<uint32, double> *>(net_ptr);

        net->clear();
        delete net;
    }

    void *build_networkdev_f64(void *host_net_ptr)
    {
        const Network_OTF<uint32, double> *h_net = static_cast<Network_OTF<uint32, double> *>(host_net_ptr);

        return upload_network<uint32, double>(h_net);
    }

    void hvec_cuda(
        void *basis_ptr,
        void *net_ptr,
        const double *__restrict__ src,
        double *__restrict__ dst)
    {
        const BasisViewDev<uint32> *basis = static_cast<BasisViewDev<uint32> *>(basis_ptr);
        const NetworkDev<uint32, double> *net = static_cast<NetworkDev<uint32, double> *>(net_ptr);

        cuda_hvec<uint32, double>(*basis, *net, src, dst);
    }

    void expm_cuda(
        void *basis_ptr,
        void *net_ptr,
        int64 idx, double theta,
        double *__restrict__ vec)
    {
        const BasisViewDev<uint32> *basis = static_cast<BasisViewDev<uint32> *>(basis_ptr);
        const NetworkDev<uint32, double> *net = static_cast<NetworkDev<uint32, double> *>(net_ptr);

        expm_svd_network_otf_gpu<uint32, double>(*basis, *net, idx, theta, vec);
    }

    void expm_regtile_cuda(
        void *basis_ptr,
        void *net_ptr,
        int64 idx, double theta,
        double *__restrict__ vec)
    {
        const BasisViewDev<uint32> *basis = static_cast<BasisViewDev<uint32> *>(basis_ptr);
        const NetworkDev<uint32, double> *net = static_cast<NetworkDev<uint32, double> *>(net_ptr);

        expm_svd_network_otf_gpu_tile<uint32, double, false>(*basis, *net, idx, theta, vec);
    }

    void expm_sharedtile_cuda(
        void *basis_ptr,
        void *net_ptr,
        int64 idx, double theta,
        double *__restrict__ vec)
    {
        const BasisViewDev<uint32> *basis = static_cast<BasisViewDev<uint32> *>(basis_ptr);
        const NetworkDev<uint32, double> *net = static_cast<NetworkDev<uint32, double> *>(net_ptr);

        expm_svd_network_otf_gpu_tile<uint32, double, true>(*basis, *net, idx, theta, vec);
    }

    double grad_cuda(
        void *basis_ptr,
        void *net_ptr,
        int64 idx, double theta,
        const double *__restrict__ lp,
        const double *__restrict__ rp)
    {
        const BasisViewDev<uint32> *basis = static_cast<BasisViewDev<uint32> *>(basis_ptr);
        const NetworkDev<uint32, double> *net = static_cast<NetworkDev<uint32, double> *>(net_ptr);

        return grad_svd_network_otf_gpu<uint32, double>(*basis, *net, idx, theta, lp, rp);
    }

    double backgrad_cuda(
        void *basis_ptr,
        void *net_ptr,
        int64 idx, double theta,
        double *__restrict__ lp,
        double *__restrict__ rp)
    {
        const BasisViewDev<uint32> *basis = static_cast<BasisViewDev<uint32> *>(basis_ptr);
        const NetworkDev<uint32, double> *net = static_cast<NetworkDev<uint32, double> *>(net_ptr);

        return backgrad_svd_network_otf_gpu<uint32, double>(*basis, *net, idx, theta, lp, rp);
    }

    void batchgrad_cuda(
        void *basis_ptr,
        void *net_ptr,
        const double *__restrict__ thetas,
        const double *__restrict__ lp,
        const double *__restrict__ rp,
        double *__restrict__ grads)
    {
        const BasisViewDev<uint32> *basis = static_cast<BasisViewDev<uint32> *>(basis_ptr);
        const NetworkDev<uint32, double> *net = static_cast<NetworkDev<uint32, double> *>(net_ptr);

        cuda_batchgrad<uint32, double>(*basis, *net, thetas, lp, rp, grads);
    }
}
