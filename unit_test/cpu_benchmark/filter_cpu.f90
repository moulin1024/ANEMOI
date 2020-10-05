module filter_cpu_m
    use precision
    use dimen
contains
subroutine filter_cpu(f,flag)

    !---------------------------------------------------------------------------
    ! declaration
    !---------------------------------------------------------------------------

    implicit none
    include '../common/fftw3.f90'

    real(fp),dimension(nx,ny,nz2),intent(inout) :: f
    integer,intent(in) :: flag

    integer :: plan_forward,plan_backward
    real(fp),dimension(nx,ny) :: f_2d
    double complex,dimension(nx/2+1,ny) :: f_hat

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
            ! filter
            do j=1,ny
                jj=j-1
                if(jj>nint(ny/2.0)) jj=jj-ny
                jj=jj*l_r
                do i=1,nx/2+1
                    ii=i-1
                    ! filter x
                    if (ii>=nint(nx/(2.0*fgr))) then
                        f_hat(i,j)=0.d0
                    ! filter y
                    elseif (abs(jj)>=nint(l_r*ny/(2.0*fgr))) then
                        f_hat(i,j)=0.d0
                    end if
                end do
            end do
            ! backward fft --> f
            call dfftw_execute_dft_c2r(plan_backward,f_hat,f(:,:,k))
        end do

    !---------------------------------------------------------------------------
    ! finalize

    elseif (flag==2) then

        call dfftw_destroy_plan(plan_forward)
        call dfftw_destroy_plan(plan_backward)

    endif

end subroutine filter_cpu

subroutine filter_sgs_cpu(f_hat,f_hatd,f,flag)

    !---------------------------------------------------------------------------
    ! declaration
    !---------------------------------------------------------------------------

    implicit none
    include '../common/fftw3.f90'

    real(fp),dimension(nx,ny,nz2),intent(out) :: f_hat,f_hatd
    real(fp),dimension(nx,ny,nz2),intent(in) :: f
    integer,intent(in) :: flag

    integer(fp) :: plan_forward,plan_backward
    real(fp),dimension(nx,ny) :: f_2d
    double complex,dimension(nx/2+1,ny) :: f_hat_ft,f_hatd_ft

    integer :: i,j,k,ii,jj

    save plan_forward,plan_backward

    !---------------------------------------------------------------------------
    ! main code
    !---------------------------------------------------------------------------

    !---------------------------------------------------------------------------
    ! init

    if (flag==0) then

       call dfftw_plan_dft_r2c_2d(plan_forward,nx,ny,f_2d,f_hat_ft,fftw_patient)
       call dfftw_plan_dft_c2r_2d(plan_backward,nx,ny,f_hat_ft,f_2d,fftw_patient)

   !---------------------------------------------------------------------------
   ! compute

   elseif (flag==1) then


        do k=1,nzb+2
            ! forward fft --> f_hat
            call dfftw_execute_dft_r2c(plan_forward,f(:,:,k),f_hat_ft)
            f_hat_ft=f_hat_ft*inxny
            f_hatd_ft=f_hat_ft
            ! filter
            do j=1,ny
                jj=j-1
                if(jj>nint(ny/2.0)) jj=jj-ny
                jj=jj*l_r
                do i=1,nx/2+1
                    ! filter x once
                    ii=i-1
                    if (ii>=nint(nx/(2.0*fgr*tfr))) then
                        f_hat_ft(i,j)=0.d0
                    ! filter y once
                    elseif (abs(jj)>=nint(l_r*ny/(2.0*fgr*tfr))) then
                        f_hat_ft(i,j)=0.d0
                    end if
                    ! filter x twice
                    if (ii>=nint(nx/(2.0*fgr*tfr*tfr))) then
                        f_hatd_ft(i,j)=0.d0
                    ! filter y twice
                    elseif (abs(jj)>=nint(l_r*ny/(2.0*fgr*tfr*tfr))) then
                        f_hatd_ft(i,j)=0.d0
                    end if
                end do
            end do
            ! backward fft --> f
            call dfftw_execute_dft_c2r(plan_backward,f_hat_ft,f_hat(:,:,k))
            call dfftw_execute_dft_c2r(plan_backward,f_hatd_ft,f_hatd(:,:,k))
        end do

    !---------------------------------------------------------------------------
    ! finalize

    elseif (flag==2) then

        call dfftw_destroy_plan(plan_forward)
        call dfftw_destroy_plan(plan_backward)

    endif

end subroutine filter_sgs_cpu

subroutine filter_wall_cpu(f_hat,f,flag)

    !---------------------------------------------------------------------------
    ! declaration
    !---------------------------------------------------------------------------

    implicit none
    include '../common/fftw3.f90'

    real(fp),dimension(nx,ny,nz2),intent(out) :: f_hat
    real(fp),dimension(nx,ny,nz2),intent(in) :: f
    integer,intent(in) :: flag

    integer :: plan_forward,plan_backward
    real(fp),dimension(nx,ny) :: f_2d
    double complex,dimension(nx/2+1,ny) :: f_hat_ft

    integer :: i,j,k,ii,jj

    save plan_forward,plan_backward

    !---------------------------------------------------------------------------
    ! main code
    !---------------------------------------------------------------------------

    !---------------------------------------------------------------------------
    ! init

    if (flag==0) then

        call dfftw_plan_dft_r2c_2d(plan_forward,nx,ny,f_2d,f_hat_ft,fftw_measure)
        call dfftw_plan_dft_c2r_2d(plan_backward,nx,ny,f_hat_ft,f_2d,fftw_measure)

    !---------------------------------------------------------------------------
    ! compute

    elseif (flag==1) then

        ! forward fft --> f_hat
        call dfftw_execute_dft_r2c(plan_forward,f,f_hat_ft)
        f_hat_ft=f_hat_ft*inxny
        ! filter
        do j=1,ny
            jj=j-1
            if(jj>nint(ny/2.0)) jj=jj-ny
            jj=jj*l_r
            do i=1,nx/2+1
                ii=i-1
                ! filter x
                if (ii>=nint(nx/(2.0*fgr*tfr))) then
                    f_hat_ft(i,j)=0.d0
                ! filter y
              elseif (abs(jj)>=nint(l_r*ny/(2.0*fgr*tfr))) then
                    f_hat_ft(i,j)=0.d0
                end if
            end do
        end do
        ! backward fft --> f
        call dfftw_execute_dft_c2r(plan_backward,f_hat_ft,f_hat)

    !---------------------------------------------------------------------------
    ! finalize

    elseif (flag==2) then

        call dfftw_destroy_plan(plan_forward)
        call dfftw_destroy_plan(plan_backward)

    endif

end subroutine filter_wall_cpu
end module filter_cpu_m