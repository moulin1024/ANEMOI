module filter_cpu
use dimen
use precision
contains

subroutine ddxy_filter_cpu(f,dfdx,dfdy,flag)

    !---------------------------------------------------------------------------
    ! declaration
    !---------------------------------------------------------------------------

    implicit none
    include '../common/fftw3.f90'

    real*8,dimension(nx,ny,nz2),intent(out) :: dfdx,dfdy
    real*8,dimension(nx,ny,nz2),intent(inout) :: f
    integer,intent(in) :: flag

    integer*8 :: plan_forward,plan_backward
    real*8,dimension(nx,ny) :: f_2d
    double complex,dimension(nx/2+1,ny) :: f_hat,dfdx_hat,dfdy_hat

    integer :: i,j,k,ii,jj

    save plan_forward,plan_backward

    !---------------------------------------------------------------------------
    ! main code
    !---------------------------------------------------------------------------

    !---------------------------------------------------------------------------
    ! init

    if (flag==0) then

       call dfftw_plan_dft_r2c_2d(plan_forward,nx,ny,f_2d,f_hat,fftw_patient)
       call dfftw_plan_dft_c2r_2d(plan_backward,nx,ny,f_hat,f_2d,fftw_patient)

    !---------------------------------------------------------------------------
    ! compute

    elseif (flag==1) then

        do k=2,nzb+1
            ! forward fft --> f_hat
            call dfftw_execute_dft_r2c(plan_forward,f(:,:,k),f_hat)
            f_hat=f_hat*inxny
            ! filter and compute deriv
            do j=1,ny
                jj=j-1
                if(jj>nint(ny/2.0)) jj=jj-ny
                jj=jj*l_r
                do i=1,nx/2+1
                    ii=i-1
                    ! filter x
                    if (ii>=nint(nx/(2.0*fgr)))then
                        f_hat(i,j)=0.d0
                    ! filter y
                    elseif(abs(jj)>=nint(l_r*ny/(2.0*fgr)))then
                        f_hat(i,j)=0.d0
                    end if
                    ! dfdx_hat
                    dfdx_hat(i,j)=dcmplx(dimag(f_hat(i,j))*(-1.d0),dreal(f_hat(i,j)))*ii
                    ! dfdy_hat
                    dfdy_hat(i,j)=dcmplx(dimag(f_hat(i,j))*(-1.d0),dreal(f_hat(i,j)))*jj
                enddo
            enddo
            ! ! backward fft --> f
            call dfftw_execute_dft_c2r(plan_backward,f_hat,f(:,:,k))
            ! backward fft --> dfdx
            call dfftw_execute_dft_c2r(plan_backward,dfdx_hat,dfdx(:,:,k))
            ! backward fft --> dfdy
            call dfftw_execute_dft_c2r(plan_backward,dfdy_hat,dfdy(:,:,k))
            
        end do
    ! print *,f_hat(3,:,10)
    !---------------------------------------------------------------------------
    ! finalize

    elseif (flag==2) then

        call dfftw_destroy_plan(plan_forward)
        call dfftw_destroy_plan(plan_backward)

    endif

end subroutine ddxy_filter_cpu
end module filter_cpu