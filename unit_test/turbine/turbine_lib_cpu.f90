module turbine_lib_cpu
    use precision
    use dimen
    contains
    function get_interp_coeff_cpu(blade_ii,blade_jj,blade_kk)
        implicit none
 
    
        real(fp),intent(in) :: blade_ii,blade_jj,blade_kk
        real(fp),dimension(3) :: blade_point
        integer*4,dimension(3) :: blade_int
        integer*4 :: i,j,k,counter
        real(fp), dimension(3,2) :: l
        real(fp), dimension(8) :: get_interp_coeff_cpu
    
        blade_point = (/blade_ii,blade_jj,blade_kk/)
        ! Get the location in the cube 
        do i = 1,3
            l(i,1) = 1._fp - (dble(blade_point(i)-int(blade_point(i))))
            l(i,2) = dble(blade_point(i)-int(blade_point(i)))
        end do
        ! Get the coefficient
        counter = 1
        do i = 1,2
            do j = 1,2
                do k = 1,2
                    get_interp_coeff_cpu(counter) = l(1,i)*l(2,j)*l(3,k)
                    counter = counter + 1
                end do
            end do
        end do 
    
    end function get_interp_coeff_cpu
    
    function compute_vel_3D_cpu(u,blade_i,blade_j,blade_k,c)
    
        implicit none
 
    
        real(fp), dimension(nx,ny,nz2),intent(in)::u
        integer*4,intent(in) :: blade_i,blade_j,blade_k
        integer*4 :: i,j,k,counter
        real(fp),dimension(8),intent(in) :: c
        real(fp),dimension(8) :: u_stencil
        real(fp) :: compute_vel_3D_cpu
        ! To be rewritten using dot_product
        counter = 1
        do i = 0,1
            do j = 0,1
                do k = 0,1
                    u_stencil(counter) = u(blade_i+i,blade_j+j,blade_k+k)
                    counter = counter + 1            
                end do
            end do
        end do 
    
        compute_vel_3D_cpu=dot_product(u_stencil,c)
    
    end function compute_vel_3D_cpu
        
    ! compute_ang_cpu
function compute_ang_cpu(CTy,CTz)

    implicit none
 

    real*8 :: CTy,CTz,gau,compute_ang_cpu,RR

    RR = dsqrt(CTy**2+CTz**2)
    if (RR <= 1E-8) then
        compute_ang_cpu = 0._fp
    else
        gau = abs(CTz/dsqrt(CTy**2+CTz**2))
        if (CTz >= 0.0)then
            if(CTy >= 0.0) compute_ang_cpu = dasin(gau)
            if(CTy < 0.0) compute_ang_cpu = PI-abs(dasin(gau))
        else
            if(CTy < 0.0) compute_ang_cpu = PI+abs(dasin(gau))
            if(CTy >= 0.0) compute_ang_cpu = 2.0_fp*PI-dasin(gau)
        end if
    end if

end function compute_ang_cpu

! compute_vel_cpu
function compute_vel_cpu(u,i,blade_j,blade_k,c1,c2,c3,c4)

    implicit none
 

    real*8, dimension(nx,ny,nz2),intent(in)::u
    integer,intent(in) :: i,blade_j,blade_k
    real*8,intent(in) :: c1,c2,c3,c4
    real*8 :: compute_vel_cpu

    compute_vel_cpu=u(i,blade_j,blade_k)*c1+ &
        u(i,blade_j+1,blade_k)*c2+ &
        u(i,blade_j,blade_k+1)*c3+ &
        u(i,blade_j+1,blade_k+1)*c4

end function compute_vel_cpu
    end module
