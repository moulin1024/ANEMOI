module divergence_cpu
    use precision
    use dimen
contains
subroutine div0_cpu(div,u,v,w)

    !---------------------------------------------------------------------------
    ! declaration
    !---------------------------------------------------------------------------
    implicit none

    real(fp),dimension(nx,ny,nz2),intent(out) :: div
    real(fp),dimension(nx,ny,nz2),intent(in) :: u,v,w

    real(fp),dimension(nx,ny,nz2) :: dudx,dvdy,dwdz
    integer :: i,j,k

    !---------------------------------------------------------------------------
    ! main code
    !---------------------------------------------------------------------------

    call ddx_cpu(dudx,u,1)
    call ddy_cpu(dvdy,v,1)
    call ddz_w_cpu(dwdz,w)

    do k=2,nzb+1
        do j=1,ny
            do i=1,nx
                div(i,j,k) = dudx(i,j,k)+dvdy(i,j,k)+dwdz(i,j,k)
            end do
        end do
    end do

end subroutine div0_cpu

subroutine divstress_uv_cpu(divt,tx,ty,tz)

    !---------------------------------------------------------------------------
    ! declaration
    !---------------------------------------------------------------------------
    implicit none

    real(fp),dimension(nx,ny,nz2),intent(out) :: divt
    real(fp),dimension(nx,ny,nz2),intent(in) :: tx,ty,tz

    real(fp),dimension(nx,ny,nz2) :: tdx,tdy,tdz
    integer :: i,j,k

    !---------------------------------------------------------------------------
    ! main code
    !---------------------------------------------------------------------------

    !---------------------------------------------------------------------------
    ! stress gradients

    call ddx_cpu(tdx,tx,1)
    call ddy_cpu(tdy,ty,1)
    call ddz_w_cpu(tdz,tz)

    !---------------------------------------------------------------------------
    ! stress divergence

    do k=2,nzb+1
        do j=1,ny
            do i=1,nx
                divt(i,j,k)=tdx(i,j,k)+ tdy(i,j,k)+ tdz(i,j,k)
            end do
        end do
    end do

end subroutine divstress_uv_cpu

subroutine divstress_w_cpu(divt,tx,ty,tz)

    !---------------------------------------------------------------------------
    ! declaration
    !---------------------------------------------------------------------------
    implicit none

    real(fp),dimension(nx,ny,nz2),intent(out) :: divt
    real(fp),dimension(nx,ny,nz2),intent(in) :: tx,ty,tz

    real(fp),dimension(nx,ny,nz2) :: tdx,tdy,tdz
    integer :: i,j,k

    !---------------------------------------------------------------------------
    ! main code
    !---------------------------------------------------------------------------

    !---------------------------------------------------------------------------
    ! stress gradients

    call ddx_cpu(tdx,tx,1)
    call ddy_cpu(tdy,ty,1)
    call ddz_uv_cpu(tdz,tz)

    !---------------------------------------------------------------------------
    ! stress divergence

    do k=2,nzb+1
        do j=1,ny
            do i=1,nx
                divt(i,j,k)=tdx(i,j,k)+ tdy(i,j,k)+ tdz(i,j,k)
            end do
        end do
    end do

    !---------------------------------------------------------------------------
    ! at the wall we have to assume that tdz(tzz)=0.0.  any better ideas?
    ! yes -> assymetric fdm ???? -> do we really care ) w(:,:,2)=0 anyway !!

    ! if (me==0) then
    !     divt(:,:,2)=tdx(:,:,2)+ tdy(:,:,2)
    ! end if

end subroutine divstress_w_cpu

subroutine ddx_cpu(dfdx,f,flag)

    !---------------------------------------------------------------------------
    ! declaration
    !---------------------------------------------------------------------------

    implicit none
    include '../common/fftw3.f90'

    real*8,dimension(nx,ny,nz2),intent(out) :: dfdx
    real*8,dimension(nx,ny,nz2),intent(in) :: f
    integer,intent(in) :: flag

    integer*8 :: plan_forward,plan_backward
    real*8,dimension(nx,ny) :: f_2d
    double complex,dimension(nx/2+1,ny) :: f_hat,dfdx_hat

    integer :: i,j,k,ii,jj

    save plan_forward,plan_backward

    !---------------------------------------------------------------------------
    ! main code
    !---------------------------------------------------------------------------

    !---------------------------------------------------------------------------
    ! init

    if (flag==0) then

        call dfftw_plan_dft_r2c_2d(plan_forward,nx,ny,f_2d,f_hat,fftw_measure)
        call dfftw_plan_dft_c2r_2d(plan_backward,nx,ny,f_hat,f_2d,fftw_measure)

    !---------------------------------------------------------------------------
    ! compute

    elseif (flag==1) then

        do k=2,nzb+1
            ! forward fft --> f_hat
            call dfftw_execute_dft_r2c(plan_forward,f(:,:,k),f_hat)
            f_hat=f_hat*inxny
            ! derivate
            do j=1,ny
                jj=j-1
                if(jj>nint(ny/2.0)) jj=jj-ny
                do i=1,nx/2+1
                    ii=i-1
                    ! filter
                    if ((ii==nint(nx/2.0)) .or. (abs(jj)>=nint(ny/2.0)))then
                        dfdx_hat(i,j)=0.d0
                    ! dfdx_hat
                    else
                        dfdx_hat(i,j)=dcmplx(dimag(f_hat(i,j))*(-1.d0),dreal(f_hat(i,j)))*ii
                    end if
                end do
            end do
            ! backward fft --> dfdx
            call dfftw_execute_dft_c2r(plan_backward,dfdx_hat,dfdx(:,:,k))

        end do

      !---------------------------------------------------------------------------
      ! finalize

      elseif (flag==2) then

            call dfftw_destroy_plan(plan_forward)
            call dfftw_destroy_plan(plan_backward)

      endif

end subroutine ddx_cpu

subroutine ddy_cpu(dfdy,f,flag)

    !---------------------------------------------------------------------------
    ! declaration
    !---------------------------------------------------------------------------

    implicit none
    include '../common/fftw3.f90'

    real*8,dimension(nx,ny,nz2),intent(out) :: dfdy
    real*8,dimension(nx,ny,nz2),intent(in) :: f
    integer, intent(in) :: flag

    integer*8 :: plan_forward,plan_backward
    real*8,dimension(nx,ny) :: f_2d
    double complex,dimension(nx/2+1,ny) :: f_hat,dfdy_hat
    
    integer :: i,j,k,ii,jj

    save plan_forward,plan_backward

    !---------------------------------------------------------------------------
    ! main code
    !---------------------------------------------------------------------------

    !---------------------------------------------------------------------------
    ! init

    if (flag==0) then

        call dfftw_plan_dft_r2c_2d(plan_forward,nx,ny,f_2d,f_hat,fftw_measure)
        call dfftw_plan_dft_c2r_2d(plan_backward,nx,ny,f_hat,f_2d,fftw_measure)

    !---------------------------------------------------------------------------
    ! compute

    elseif (flag==1) then

        do k=2,nzb+1
            ! forward fft --> f_hat
            call dfftw_execute_dft_r2c(plan_forward,f(:,:,k),f_hat)
            f_hat=f_hat*inxny
            ! derivate
            do j=1,ny
                jj=j-1
                if(jj>nint(ny/2.0)) jj=jj-ny
                jj=jj*l_r
                do i=1,nx/2+1
                    ii=i-1
                    ! filter
                    if ((ii==nint(nx/2.0)) .or. (abs(jj)==nint(l_r*ny/2.0)))then
                        dfdy_hat(i,j)=0.d0
                    ! dfdx_hat
                    else
                        dfdy_hat(i,j)=dcmplx(dimag(f_hat(i,j))*(-1.d0),dreal(f_hat(i,j)))*jj
                    end if
                end do
            end do
            ! backward fft --> dfdx
            call dfftw_execute_dft_c2r(plan_backward,dfdy_hat,dfdy(:,:,k))

        end do

    !---------------------------------------------------------------------------
    ! finalize

    elseif (flag==2) then

        call dfftw_destroy_plan(plan_forward)
        call dfftw_destroy_plan(plan_backward)

    endif

end subroutine ddy_cpu

pure subroutine ddz_uv_cpu (dfdz,f)

    !---------------------------------------------------------------------------
    ! declaration
    !---------------------------------------------------------------------------

    implicit none

    real(fp),dimension(nx,ny,nz2),intent(out) :: dfdz
    real(fp),dimension(nx,ny,nz2),intent(in) :: f

    integer :: i,j,k

    !---------------------------------------------------------------------------
    ! main code
    !---------------------------------------------------------------------------

    do k=2,nzb+1
        do j=1,ny
            do i=1,nx
                dfdz(i,j,k)=(f(i,j,k)-f(i,j,k-1))*idz
            end do
        end do
    end do

end subroutine ddz_uv_cpu

pure subroutine ddz_w_cpu (dfdz,f)

    !---------------------------------------------------------------------------
    ! declaration
    !---------------------------------------------------------------------------

    implicit none

    real(fp),dimension(nx,ny,nz2),intent(out) :: dfdz
    real(fp),dimension(nx,ny,nz2),intent(in) :: f

    integer :: i,j,k

    !---------------------------------------------------------------------------
    ! main code
    !---------------------------------------------------------------------------

    do k=2,nzb+1
        do j=1,ny
            do i=1,nx
                dfdz(i,j,k)=(f(i,j,k+1)-f(i,j,k))*idz
            end do
        end do
    end do

end subroutine ddz_w_cpu

pure subroutine ddz_uv_cpu_corr (dfdz,k_corr)

    !---------------------------------------------------------------------------
    ! declaration
    !---------------------------------------------------------------------------

    implicit none

    real(fp),dimension(nx,ny,nz2),intent(inout):: dfdz
    integer,intent(in) :: k_corr

    real(fp) :: d_avg1

    integer :: i,j

    real(fp),parameter :: fr1 = 1.d0/dlog(3.d0)-1.d0

    !---------------------------------------------------------------------------
    ! main code
    !---------------------------------------------------------------------------

    d_avg1 = 0.d0

    do j=1,ny
        do i=1,nx
            d_avg1= d_avg1 + dfdz(i,j,k_corr)
        end do
    end do

    d_avg1 = d_avg1*inxny

    do j=1,ny
        do i=1,nx
            dfdz(i,j,k_corr)= dfdz(i,j,k_corr) + fr1*d_avg1
        end do
    end do

end subroutine ddz_uv_cpu_corr

end module divergence_cpu