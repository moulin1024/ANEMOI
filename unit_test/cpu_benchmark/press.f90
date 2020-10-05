!###############################################################################
! press: compute pressure with direct poisson solver.
!###############################################################################
! 1. press (call tridiag)
! 2. tridiag
!###############################################################################

!===============================================================================
! 1. press
!===============================================================================
module press_cpu
    use precision
    use dimen
    contains
subroutine press(p,rhs_p,dpdx,dpdy,me,nall,flag)

    !---------------------------------------------------------------------------
    ! declaration
    !---------------------------------------------------------------------------

    implicit none
    include '../common/fftw3.f90'
    integer :: me,nall
    ! routine output/input
    real(fp),dimension(nx,ny,nz2),intent(out) :: p,dpdx,dpdy
    real(fp),dimension(nx,ny,nz2),intent(in) :: rhs_p
    integer,intent(in) :: flag

    ! x-slab parallelization
    integer*4 :: me_x,me_y,i_global,j_global
    integer,parameter :: nprocs_x=1
    integer,parameter :: nprocs_y=nprocs
    integer,parameter :: nxb=(nx/2+1)/nprocs_x
    integer,parameter :: nyb=ny/nprocs_y
    integer*4,parameter :: lsize=nxb*nyb*nzb

    ! tridiag
    real(fp),dimension(nz+1) :: rhs_col,a1,b1,c1,p_colr,p_coli

    ! fftw
    integer*8 :: plan_forward,plan_backward
    real(fp),dimension(nx,ny) :: f_2d
    double complex,dimension(nx/2+1,ny) :: h_x_2d,p_hat,dpdx_hat,dpdy_hat
    double complex,dimension(nx/2+1,ny,nzb) :: h_x,h_z
    double complex,dimension(nxb,nyb,nz) :: h_2,h_3

    ! iterators
    integer :: i,j,k,ii,jj

    ! extended vertical size
    integer,parameter :: size=nz+1

    save plan_forward,plan_backward

    !---------------------------------------------------------------------------
    ! main code
    !---------------------------------------------------------------------------

    !---------------------------------------------------------------------------
    ! init

    if (flag == 0) then

        call dfftw_plan_dft_r2c_2d(plan_forward,nx,ny,f_2d,p_hat,fftw_patient)
        call dfftw_plan_dft_c2r_2d(plan_backward,nx,ny,p_hat,f_2d,fftw_patient)

    !---------------------------------------------------------------------------
    ! compute

    elseif (flag == 1) then

        !-----------------------------------------------------------------------
        ! forward fft --> h_x (fourrier transform of rhs_p)

        ! interior
        do k=2,nzb+1
            call dfftw_execute_dft_r2c(plan_forward,rhs_p(:,:,k),h_x_2d)
            h_x(:,:,k-1)=h_x_2d*inxny
        end do


        !-----------------------------------------------------------------------
        ! handle x-slab parallelization (1)

        ! redistribute parallelization from h_x to h_3 to h_2
        do j=1,nprocs_y
            do i=1,nprocs_x
                k=(j-1)*nprocs_x+i-1
                if (k==me) then
                    me_x = i
                    me_y = j
                end if
                h_3(:,:,k*nzb+1:(k+1)*nzb)= h_x((i-1)*nxb+1:i*nxb,(j-1)*nyb+1:j*nyb,1:nzb)
            end do
        end do


        call mpi_alltoall(h_3(1,1,1),lsize,mpi_double_complex, &
            h_2(1,1,1),lsize,mpi_double_complex,nall,ierr)

        !-----------------------------------------------------------------------
        ! solve tridiag

        if(me<(nprocs_x*nprocs_y))then
            do j=1,nyb

                ! this is were we have to modify !!! wavenumber already computed !!
                ! l_r = 2*pi/Ly --> jj = 2*pi*ky/Ly
                j_global=j + (me_y-1)*nyb
                jj=j_global-1
                if(jj>(ny/2)) jj=jj-ny
                jj=jj*l_r
                do i=1,nxb

                    i_global=i + (me_x-1)*nxb
                    ii=i_global-1

                    !**** real part ****

                    ! assemble bot
                    rhs_col(1)=0.d0
                    a1(1)=0.d0
                    b1(1)=-1.d0
                    c1(1)=1.d0
                    if ((ii==0).and.(jj==0)) then
                        rhs_col(1)=0.d0
                        a1(1)=0.d0
                        b1(1)=1.d0
                        c1(1)=0.d0
                    end if

                    ! assemble interior
                    do k=2,nz
                        rhs_col(k)=dreal(h_2(i,j,k-1))
                        a1(k)=1.d0/(dz**2)
                        b1(k)=(-ii*ii-jj*jj-2.d0/(dz**2))
                        c1(k)=1.d0/(dz**2)
                    end do

                    ! assemble top
                    rhs_col(nz+1)=0.d0
                    a1(nz+1)= -1.d0
                    b1(nz+1)=  1.d0
                    c1(nz+1)=  0.d0

                    ! solve tridiag
                    call tridiag_cpu (a1,b1,c1,rhs_col,p_colr,size)

                    !**** imag part ****

                    ! assemble bot
                    rhs_col(1)=0.d0
                    a1(1)=  0.d0
                    b1(1)= -1.d0
                    c1(1)=  1.d0
                    if ((ii==0).and.(jj==0)) then
                        a1(1)=0.d0
                        b1(1)=1.d0
                        c1(1)=0.d0
                        rhs_col(1)=0.d0
                    end if

                    ! assemble interior
                    do k=2,nz
                        rhs_col(k)=dimag(h_2(i,j,k-1))
                        a1(k)=1.d0/(dz**2)
                        b1(k)=(-ii*ii-jj*jj-2.d0/(dz**2))
                        c1(k)=1.d0/(dz**2)
                    end do

                    ! assemble top
                    rhs_col(nz+1)=0.d0
                    b1(nz+1)=  1.d0
                    a1(nz+1)= -1.d0
                    c1(nz+1)=  0.d0

                    ! solve tridiag
                    call tridiag_cpu (a1,b1,c1,rhs_col,p_coli,size)

                    !**** real + imag ****

                    do k=1,nz
                        h_2(i,j,k)=dcmplx(p_colr(k+1),p_coli(k+1))
                    end do

                end do
            end do
        end if

        !-----------------------------------------------------------------------
        ! handle x-slab parallelization (2)

        ! alltoall from h_2 to h_3 to h_z
        call mpi_alltoall(h_2(1,1,1),lsize,mpi_double_complex, &
            h_3(1,1,1),lsize,mpi_double_complex,nall,ierr)

    

        do j=1,nprocs_y
            do i=1,nprocs_x
                k=(j-1)*nprocs_x+i-1
                h_z((i-1)*nxb+1:i*nxb,(j-1)*nyb+1:j*nyb,1:nzb)=h_3(:,:,k*nzb+1:(k+1)*nzb)
            end do
        end do
        !-----------------------------------------------------------------------
        ! compute p, dpdx, dpdy

        do k=2,nzb+1

            ! notice the k-1
            p_hat=h_z(:,:,k-1)
            
            do j=1,ny
                jj=j-1
                if(jj>nint(ny/2.0)) jj=jj-ny
                jj=jj*l_r
                do i=1,nx/2+1
                    ii=i-1
                    ! filter x
                    if (ii>=nint(nx/(2.0)))then
                    ! if (ii>=nint(nx/(2.0*fgr)))then
                        p_hat(i,j)=0.d0
                    ! filter y
                    elseif(abs(jj)>=nint(l_r*ny/(2.0)))then
                    ! elseif(abs(jj)>=nint(l_r*ny/(2.0*fgr)))then
                        p_hat(i,j)=0.d0
                    end if
                    ! dpdx_hat
                    dpdx_hat(i,j)=dcmplx(dimag(p_hat(i,j))*(-1.d0),dreal(p_hat(i,j)))*ii
                    ! dpdy_hat
                    dpdy_hat(i,j)=dcmplx(dimag(p_hat(i,j))*(-1.d0),dreal(p_hat(i,j)))*jj
                end do
            end do
            
            ! backward fft --> p
            call dfftw_execute_dft_c2r(plan_backward,p_hat,p(:,:,k))
            ! backward fft --> dpdx

            ! if(k == 19) print *,me,dpdx_hat(:,17)
            call dfftw_execute_dft_c2r(plan_backward,dpdx_hat,dpdx(:,:,k))
            ! backward fft --> dpdy
            call dfftw_execute_dft_c2r(plan_backward,dpdy_hat,dpdy(:,:,k))

        end do

    !---------------------------------------------------------------------------
    ! finalize

    elseif (flag == 2) then

        call dfftw_destroy_plan(plan_forward)
        call dfftw_destroy_plan(plan_backward)

    endif

end subroutine press

!===============================================================================
! 2. tridiag
!===============================================================================

SUBROUTINE tridiag_cpu(a,b,c,r,u,n)

      INTEGER*4 n,NMAX
      real(fp) a(n),b(n),c(n),r(n),u(n)
      PARAMETER (NMAX=500)
      INTEGER*4 j
      real(fp) bet,gam(NMAX)

      if(b(1)==0.)pause 'tridiag: rewrite equations'
      bet=b(1)
      u(1)=r(1)/bet
      do 11 j=2,n
        gam(j)=c(j-1)/bet
        bet=b(j)-a(j)*gam(j)
        if(bet==0.) then
           print *, 'tridiag failed at k=',j
           print *,'a, b, c, gam, and bet=',a(j),b(j),c(j),gam(j),bet
           pause
        end if
        u(j)=(r(j)-a(j)*u(j-1))/bet
11    continue
      do 12 j=n-1,1,-1
        u(j)=u(j)-gam(j+1)*u(j+1)
12    continue

      return

END SUBROUTINE tridiag_cpu
! (C) Copr. 1986-92 Numerical Recipes Software ]2#"0>Ya%.
end module press_cpu