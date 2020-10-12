!===============================================================================
! compute turbine
!===============================================================================
! warning: chord and twist are hardcoded for wire01
module turbine_cpu_m

    use precision
    use dimen

    real(fp), parameter :: dang = 2.0_fp*pi/real(n_phi)
    real(fp), dimension(n_turb), parameter :: dr = [(turb_r(idx1)/real(n_r), idx1=1,n_turb)]
    real(fp), dimension(n_r,n_turb), parameter :: r  = reshape([((real(idx2-0.5_fp)*dr(idx1), idx2=1,n_r), idx1=1,n_turb)],(/n_r,n_turb/))
    real(fp), dimension(n_r,n_turb), parameter :: dA = reshape([((r(idx2,idx1)*dang*dr(idx1), idx2=1,n_r), idx1=1,n_turb)],(/n_r,n_turb/))

    real(fp), dimension(n_phi), parameter :: cos_Tang = [(cos(dble(i_phi-1)*dang), i_phi=1,n_phi)]
    real(fp), dimension(n_phi), parameter :: sin_Tang = [(sin(dble(i_phi-1)*dang), i_phi=1,n_phi)]

    integer,parameter :: ker_x_width = 16
    integer,parameter :: ker_y_width = 32

    contains
Subroutine turbine_cpu(Fx,Fy,Fz,thrust,power,sum_ker_u,sum_ker_w,&
                        CTFx,CTFt,CTux,CTaoa,omega,u,v,w,t,me,nall)

    !---------------------------------------------------------------------------
    ! declaration
    !---------------------------------------------------------------------------

    implicit none
 
    real(fp), dimension(n_phi,n_r,n_turb) :: sum_ker_u,sum_ker_w
    real(fp),dimension(nx,ny,nz2),intent(out) :: fx,fy,fz
    real(fp),intent(inout) :: thrust(n_turb),power(n_turb),omega(n_turb)
    real(fp), dimension(n_phi,n_r,n_turb),intent(inout) :: CTFx,CTFt,CTux,CTaoa
    real(fp),dimension(nx,ny,nz2),intent(in) :: u,v,w
    integer*4,intent(in) :: t,me,nall
    real(fp), dimension(n_turb,3) :: wt_coord_yaw
    integer :: i,j,k,jj,kk
    real(fp), dimension(nx,n_turb) :: ker_x
    real(fp) :: sigma,tmp1,tmp2,tmp3,sum_ker_x

    save ker_x,wt_coord_yaw

    !---------------------------------------------------------------------------
    ! main code
    !---------------------------------------------------------------------------

    !---------------------------------------------------------------------------
    ! init

    fx = 0.d0
    fy = 0.d0
    fz = 0.d0
    thrust = 0.d0
    power = 0.d0

    !---------------------------------------------------------------------------
    ! pre-process x-project

    if (t==1) then
        if (turb_flag > 0) then
            open(unit=199,file='input/wt_coord.csv')
            read(199,*)
            do k=1,n_turb
                read(199,*) wt_coord_yaw(k,1:3)
            end do
            close(199)
        end if
    endif


    !---------------------------------------------------------------------------
    ! compute models

    ! if(tower_model == 1)then
    !     ! call tower(Fx,Fy,Fz,thrust,power,u,v,w,t,me,nall,wt_coord_yaw)
    ! end if

    ! if(nacelle_model == 1)then
    !     ! call nacelle(Fx,Fy,Fz,thrust,power,u,v,w,t,me,nall,wt_coord_yaw)
    ! end if
    
    ! if(turbine_model == 2)then
        call admryaw_cpu(Fx,Fy,Fz,thrust,power,sum_ker_u,sum_ker_w,CTFx,CTFt,omega,u,v,w,t,me,nall,wt_coord_yaw)
    ! end if

end Subroutine turbine_cpu


!===============================================================================
! compute ADMR for yaw turbine
!===============================================================================

Subroutine admryaw_cpu(Fx,Fy,Fz,thrust,power,sum_ker_u,sum_ker_w,CTFx,CTFt,omega,u,v,w,t,me,nall,wt_coord_yaw)

    !---------------------------------------------------------------------------
    ! declaration
    !---------------------------------------------------------------------------
    use turbine_lib_cpu
    implicit none
 
    integer :: i_phi,i_r,i_turb
    real(fp), dimension(nx,ny,nz2), intent(in) :: u,v,w
    real(fp), dimension(nx,ny,nz2), intent(inout) ::fx,fy,fz
    real(fp), intent(inout) :: thrust(n_turb),power(n_turb)
    integer*4, intent(in) :: t,me,nall

    real(fp), dimension(n_turb,3) :: wt_coord_yaw
    integer,parameter :: ker_x_width = 16
    integer,parameter :: ker_y_width = 32
    integer*4 :: i,j,k,me_k,index,x_index,hub_node
    real(fp), dimension(nx,ny,nz2) :: u_,v_,w_,fx_,fy_,fz_
    real(fp), dimension(ny,nz2,ker_x_width,n_turb):: CTx,CTy,CTzu,CTzw,CTru,CTrw
    real(fp), dimension(n_phi,n_r,n_turb) :: sum_ker_u,sum_ker_w
    real(fp), dimension(n_phi,n_r,n_turb) :: CTFx,CTFt

    real(fp) :: blade_ii,blade_jj,blade_kk_w,blade_kk_uv
    real(fp) :: mu_x,mu_y,mu_z,sigma,sigma_x,tmp1,tmp2,tmp3,sum_ker_x
    real(fp) :: omega(n_turb),CTphi,sin_CTphi,cos_CTphi
    real(fp) :: CTU1,CTV,CTW,CTV1,CTUrel,CTUd,AoA
    real(fp) :: CTCL,CTCD,CTFy,CTFx_origin,CTFy_origin,CTU1_origin,CTV_origin,CTW_origin
    real(fp) :: f1,f2
    real(fp) :: chordl
    real(fp) :: lx1,lx2,ly1,ly2,lz1_w,lz2_w,lz1_uv,lz2_uv

    ! fct
    ! real(fp), external :: compute_ang_cpu,compute_vel_cpu,rpm_wire01!,get_interp_coeff_cpu
    ! csts
    ! real(fp), dimension(n_r,n_turb), parameter :: dA = [((r(i_r,i_turb)*dang*dr(i_turb), i_r=1,n_r), i_turb=1,n_turb)]

    real(fp), dimension(n_phi), parameter :: cos_Tang = [(cos(dble(i_phi-1)*dang), i_phi=1,n_phi)]
    real(fp), dimension(n_phi), parameter :: sin_Tang = [(sin(dble(i_phi-1)*dang), i_phi=1,n_phi)]
    
    real(fp), dimension(:), allocatable :: radius_dat != (/0.0,0.0075,0.0125,0.0175,0.0226,0.0275,0.0325,0.0375,0.0425,0.0476,0.0525,0.0575,0.0625,0.0675,0.0726,0.075/)
    real(fp), dimension(:), allocatable :: twist_dat != (/0.7016,0.7016,0.5525,0.4389,0.3559,0.2949,0.2458,0.2085,0.1797,0.1542,0.1339,0.1187,0.1051,0.0932,0.0831,0.0831/)*rad2deg
    real(fp), dimension(:), allocatable :: chord_dat !=(/0.0138,0.0138,0.0163,0.0176,0.0186,0.0195,0.0193,0.0174,0.0155,0.0141,0.0127,0.0115,0.0104,0.0096,0.0088,0.0088/)

    ! real(fp), dimension(16), parameter :: radius_dat= (/0.0,0.0075,0.0125,0.0175,0.0226,0.0275,0.0325,0.0375,0.0425,0.0476,0.0525,0.0575,0.0625,0.0675,0.0726,0.075/)
    ! real(fp), dimension(16), parameter :: twist_dat = (/0.7016,0.7016,0.5525,0.4389,0.3559,0.2949,0.2458,0.2085,0.1797,0.1542,0.1339,0.1187,0.1051,0.0932,0.0831,0.0831/)*rad2deg
    ! real(fp), dimension(16), parameter :: chord_dat =(/0.0138,0.0138,0.0163,0.0176,0.0186,0.0195,0.0193,0.0174,0.0155,0.0141,0.0127,0.0115,0.0104,0.0096,0.0088,0.0088/)


    real(fp),dimension(n_r,n_turb) :: twist,CTsoli
    real(fp), dimension(ny,nz2,ker_x_width,n_turb) :: sin_CTangx,cos_CTangz
    real(fp), dimension(n_phi,n_r,ny,nz2,ker_x_width,n_turb):: ker_u,ker_w
    real(fp), dimension(ny) :: flag_j

    real(fp) :: inflow_window(n_turb,10),inflow_inst,inflow_smooth(n_turb),inflow_tmp
    integer, dimension(n_phi,n_r,n_turb) :: blade_i, blade_j,blade_k_w,blade_k_uv,flag_me
    real(fp), dimension(n_phi,n_r,n_turb) :: c1_w,c2_w,c3_w,c4_w,c5_w,c6_w,c7_w,c8_w,c1_uv,c2_uv,c3_uv,c4_uv,c5_uv,c6_uv,c7_uv,c8_uv

    integer io,nlines

    real(fp), dimension(3,181) :: alpha_cl_cd
    real(fp),dimension(:,:),allocatable :: turbine_geo

    real(fp), dimension(8) :: c_w_1d,c_uv_1d
    real(fp), dimension(n_phi,n_r,n_turb,8) :: c_w,c_uv
    save sin_CTangx,cos_CTangz,twist,CTsoli,ker_u,ker_w,flag_j, &
        blade_j,blade_k_w,blade_k_uv,flag_me,c_w,c_uv,me_k,inflow_smooth


    !---------------------------------------------------------------------------
    ! pre-process for interp and project
    !---------------------------------------------------------------------------
    ! do i_turb=1,n_turb
    !     turb_x(i_turb) = wt_coord_yaw(i_turb,1)/z_i
    !     turb_y(i_turb) = wt_coord_yaw(i_turb,2)/z_i
    !     yaw_angle(i_turb) = wt_coord_yaw(i_turb,3)/180*pi
    ! end do

    if (t == 1) then

            open(unit=99,file='input/cl_cd.dat')
            do k=1,181
                read(99,*) alpha_cl_cd(1:3,k)
            enddo
            close(99)

        open(unit=100,file='input/turbine_geo.dat')
        ! Get the line number by read the file twice....any better idea?
        nlines = 0
        DO
            READ(100,*,iostat=io)
            IF (io/=0) EXIT
            nlines = nlines + 1
        END DO
        close(100)
        ! print *,nlines
        ! stop

        allocate(radius_dat(nlines))
        allocate(twist_dat(nlines))
        allocate(chord_dat(nlines))
        allocate(turbine_geo(3,nlines))

        open(unit=101,file='input/turbine_geo.dat')

        DO k=1,nlines
            READ(101,*,iostat=io) turbine_geo(1:3,k)
            IF (io/=0) EXIT
        END DO
        close(101)
        
        radius_dat = turbine_geo(1,:)
        twist_dat = turbine_geo(2,:)
        chord_dat = turbine_geo(3,:)
        
        ! do i_turb=1,n_turb
        !     turb_x(i_turb) = wt_coord_yaw(i_turb,1)/z_i
        !     turb_y(i_turb) = wt_coord_yaw(i_turb,2)/z_i
        !     yaw_angle(i_turb) = wt_coord_yaw(i_turb,3)/180*pi
        ! end do

        

        ker_u = 0.d0
        ker_w = 0.d0
        sum_ker_u  = 0.0
        sum_ker_w  = 0.0
        flag_j = 0
        flag_me = 0

        do i_turb=1,n_turb
            do i = 1,ker_x_width
                do k=2,nzb+1
                    do j=int(turb_y(i_turb)/dy)-int(0.5_fp*ker_y_width)+1,int(turb_y(i_turb)/dy)+int(0.5_fp*ker_y_width)
                        ! x-distance to the hub
                        CTx(j,k,i,i_turb) = (i-0.5_fp*(ker_x_width))*dx
                        ! y-distance to the hub
                        ! CTy(j,k,i,i_turb)    = (turb_j(i_turb)-j)*dy
                        ! print *,(turb_j(i_turb)-j)*dy,turb_y(i_turb)-j*dy
                        CTy(j,k,i,i_turb)    = turb_y(i_turb)-j*dy
                        ! z-distance to the hub for stagger u,v and w node
                        CTzu(j,k,i,i_turb)   = (me*nzb+k-1.5d0)*dz-turb_z(i_turb)
                        CTzw(j,k,i,i_turb)   = CTzu(j,k,i,i_turb)-0.5d0*dz
                        ! radial distance to the hub
                        CTru(j,k,i,i_turb)   = sqrt(CTy(j,k,i,i_turb)**2.+CTzu(j,k,i,i_turb)**2.)
                        CTrw(j,k,i,i_turb)   = sqrt(CTy(j,k,i,i_turb)**2.+CTzw(j,k,i,i_turb)**2.)
                        ! angle of the blade element
                        sin_CTangx(j,k,i,i_turb) = sin(compute_ang_cpu(CTy(j,k,i,i_turb),CTzu(j,k,i,i_turb)))
                        cos_CTangz(j,k,i,i_turb) = cos(compute_ang_cpu(CTy(j,k,i,i_turb),CTzw(j,k,i,i_turb)))

                    end do
                end do
            end do

            do i_r=n_r_s,n_r

                ! WiRE-01
                call itp1D_cpu( radius_dat,chord_dat,size(radius_dat),r(i_r,i_turb)*z_i,chordl )
                chordl = chordl/z_i
                call itp1D_cpu( radius_dat,twist_dat,size(radius_dat),r(i_r,i_turb)*z_i,twist(i_r,i_turb) )
                CTsoli(i_r,i_turb)  = 3.d0*chordl/(2.d0*pi*r(i_r,i_turb))

            enddo

            do i_r=n_r_s,n_r
                do i_phi=1,n_phi

                    !-----------------------------------------------------------
                    ! pre-process for interp

                    blade_kk_w = dble(turb_z(i_turb)/dz)+r(i_r,i_turb)*sin_Tang(i_phi)/dz
                    me_k=blade_kk_w/Nzb

                    if (me == me_k) then

                        flag_me(i_phi,i_r,i_turb) = 1

                        ! blade_ii = dble(turb_i(i_turb))+r(i_r,i_turb)&
                        !             *dsin(yaw_angle(i_turb))*cos_Tang(i_phi)/dx
                        ! blade_jj = dble(turb_j(i_turb))-r(i_r,i_turb)&
                        !             *dcos(yaw_angle(i_turb))*cos_Tang(i_phi)/dy
                        blade_ii = dble(turb_x(i_turb)/dx)+r(i_r,i_turb)&
                        *dsin(yaw_angle(i_turb))*cos_Tang(i_phi)/dx
                        blade_jj = dble(turb_y(i_turb)/dy)-r(i_r,i_turb)&
                        *dcos(yaw_angle(i_turb))*cos_Tang(i_phi)/dy
                        blade_kk_w = blade_kk_w-Nzb*me+2.d0
                        blade_kk_uv = blade_kk_w-0.5d0

                        blade_i(i_phi,i_r,i_turb) = int(blade_ii)
                        blade_j(i_phi,i_r,i_turb) = int(blade_jj)
                        blade_k_w(i_phi,i_r,i_turb) = int(blade_kk_w)
                        blade_k_uv(i_phi,i_r,i_turb) = int(blade_kk_uv)

                        c_uv_1d = get_interp_coeff_cpu(blade_ii,blade_jj,blade_kk_uv)
                        c_w_1d = get_interp_coeff_cpu(blade_ii,blade_jj,blade_kk_w)
                        ! Assign 1d array to multi-dim array
                        do index = 1,8
                            c_uv(i_phi,i_r,i_turb,index) = c_uv_1d(index)
                            c_w(i_phi,i_r,i_turb,index) = c_w_1d(index)
                        end do                        
                    endif

                    !-----------------------------------------------------------
                    ! pre-process for project
                    mu_x = r(i_r,i_turb)*cos_Tang(i_phi)*dsin(yaw_angle(i_turb)) 
                    mu_y = r(i_r,i_turb)*cos_Tang(i_phi)*dcos(yaw_angle(i_turb))
                    mu_z = r(i_r,i_turb)*sin_Tang(i_phi)
                    
                    sigma_x = 2.0*dx
                    sigma = 1.d0*dsqrt((r(i_r,i_turb)*dang)**2.d0+dr(i_turb)**2.d0)
                    ! 3D Gaussian kernel
                    tmp1  = 1.d0/(sigma**2*sigma_x*dsqrt(pi)**3.d0)
                    tmp2  = -1.d0!/sigma**2.d0

                    do i = 1,ker_x_width                            
                        do k=2,nzb+1 
                            do j=int(turb_y(i_turb)/dy)-int(0.5_fp*ker_y_width)+1,int(turb_y(i_turb)/dy)+int(0.5_fp*ker_y_width)
                                tmp3 = ((CTx(j,k,i,i_turb)-mu_x)/sigma_x)**2. &
                                        +((CTy(j,k,i,i_turb)-mu_y)/sigma)**2.&
                                        +((CTzu(j,k,i,i_turb)-mu_z)/sigma)**2.
                                ! if ( sqrt(tmp3) <= 4.0) then
                                    ker_u(i_phi,i_r,j,k,i,i_turb) = tmp1*dexp(tmp2*tmp3)
                                    sum_ker_u(i_phi,i_r,i_turb) = sum_ker_u(i_phi,i_r,i_turb)+ker_u(i_phi,i_r,j,k,i,i_turb)
                                ! end if

                                tmp3 = ((CTx(j,k,i,i_turb)-mu_x)/sigma_x)**2. &
                                        +((CTy(j,k,i,i_turb)-mu_y)/sigma)**2.&
                                        +((CTzw(j,k,i,i_turb)-mu_z)/sigma)**2.
                                ! if ( sqrt(tmp3) <= 4.0) then
                                    ker_w(i_phi,i_r,j,k,i,i_turb) = tmp1*dexp(tmp2*tmp3)
                                    sum_ker_w(i_phi,i_r,i_turb) = sum_ker_w(i_phi,i_r,i_turb)+ker_w(i_phi,i_r,j,k,i,i_turb)
                                ! end if
                            end do
                        end do
                    end do

                end do
            end do
        enddo

        ! call mpi_allreduce(mpi_in_place, sum_ker_u(1,1,1),&
        !     n_phi*n_r*n_turb,mpi_double_precision,&
        !     mpi_sum,nall,ierr)

        ! call mpi_allreduce(mpi_in_place, sum_ker_w(1,1,1),&
        !     n_phi*n_r*n_turb,mpi_double_precision,&
        !     mpi_sum,nall,ierr)

        do i_turb=1,n_turb
            do i = 1,ker_x_width
                do k=2,Nzb+1
                    do j=int(turb_y(i_turb)/dy)-int(0.5_fp*ker_y_width)+1,int(turb_y(i_turb)/dy)+int(0.5_fp*ker_y_width)
                        do i_r=n_r_s,n_r
                            do i_phi=1,n_phi

                                ker_u(i_phi,i_r,j,k,i,i_turb) = ker_u(i_phi,i_r,j,k,i,i_turb)&
                                                                /sum_ker_u(i_phi,i_r,i_turb)
                                ker_w(i_phi,i_r,j,k,i,i_turb) = ker_w(i_phi,i_r,j,k,i,i_turb)&
                                                                /sum_ker_w(i_phi,i_r,i_turb)
                                if ( ker_u(i_phi,i_r,j,k,i,i_turb)>= 1E-15 .or. &
                                    ker_w(i_phi,i_r,j,k,i,i_turb)>= 1E-15) then
                                    flag_j(j)=1
                                endif

                            end do
                        end do
                    end do
                end do
            end do
        end do

    end if

    !---------------------------------------------------------------------------
    ! compute
    !---------------------------------------------------------------------------

    !---------------------------------------------------------------------------
    ! interpolate from cartesian grids to BE

    ctfx =0.d0
    ctft =0.d0

    !---------------------------------------------------------------------------
    ! get the hub-height inflow velocity
    ! if (me == me_k) then   
        
    !     ! do i_turb=1,n_turb 
    !     !     hub_node = int(turb_z(i_turb)/dz)-Nzb*me_k+2.d0

    !     !     inflow_inst = sum(u(nint(turb_x(i_turb)/dx-8*turb_r(i_turb)/dx),&
    !     !                            nint(turb_y(i_turb)/dy-5*turb_r(i_turb)/dy):&
    !     !                            nint(turb_y(i_turb)/dy+5*turb_r(i_turb)/dy),hub_node))/&
    !     !                     size(u(nint(turb_x(i_turb)/dx-8*turb_r(i_turb)/dx),&
    !     !                            nint(turb_y(i_turb)/dy-5*turb_r(i_turb)/dy):&
    !     !                            nint(turb_y(i_turb)/dy+5*turb_r(i_turb)/dy),hub_node))
            
    !     !     ! print *,u(nint(turb_x(i_turb)/dx-8*turb_r(i_turb)/dx),&
    !     !     ! nint(turb_y(i_turb)/dy),hub_node)
    !     !     inflow_window(i_turb,mod((t-1)/10,10)+1) = u(nint(turb_x(i_turb)/dx-8*turb_r(i_turb)/dx),&
    !     !     nint(turb_y(i_turb)/dy),hub_node)
    !     !     inflow_tmp = sum(inflow_window(i_turb,:))/10
    !     !     print *,inflow_tmp,mod((t-1)/10,10)+1
    !     !     ! inflow_smooth = sum(inflow_window,2)/10*0.0
    !     !     ! print *,inflow_smooth
    !     ! end do
    ! end if
    ! call MPI_Bcast(inflow_smooth, n_turb ,mpi_double_precision, me_k, MPI_COMM_WORLD,ierr)

    do i_turb=1,n_turb
        omega(i_turb) = 253._fp!3.8*inflow(i_turb)/(turb_r(i_turb)*z_i)
        !rpm_wire01(inflow_smooth(i_turb))/60*2*pi!3.8*inflow(i_turb)/(turb_r(i_turb)*z_i)
        do i_r=n_r_s,n_r
            do i_phi=1,n_phi
                if (flag_me(i_phi,i_r,i_turb) == 1) then

                    !-----------------------------------------------------------
                    ! ref vel --> we intepolate at BE center

                    if (nprocs==1 .or. me<nprocs/2) then

                        CTU1_origin=compute_vel_3D_cpu(u,blade_i(i_phi,i_r,i_turb),&
                            blade_j(i_phi,i_r,i_turb),blade_k_uv(i_phi,i_r,i_turb),&
                            pack(c_uv(i_phi,i_r,i_turb,:),.true.))

                        CTV_origin=compute_vel_3D_cpu(v,blade_i(i_phi,i_r,i_turb),&
                            blade_j(i_phi,i_r,i_turb),blade_k_uv(i_phi,i_r,i_turb),&
                            pack(c_uv(i_phi,i_r,i_turb,:),.true.))

                        CTW_origin=compute_vel_3D_cpu(w,blade_i(i_phi,i_r,i_turb),&
                            blade_j(i_phi,i_r,i_turb),blade_k_w(i_phi,i_r,i_turb),&
                            pack(c_w(i_phi,i_r,i_turb,:),.true.))

                    else

                        CTU1_origin=compute_vel_3D_cpu(u_,blade_i(i_phi,i_r,i_turb),&
                            blade_j(i_phi,i_r,i_turb),blade_k_uv(i_phi,i_r,i_turb),&
                            pack(c_uv(i_phi,i_r,i_turb,:),.true.))

                        CTV_origin=compute_vel_3D_cpu(v_,blade_i(i_phi,i_r,i_turb),&
                            blade_j(i_phi,i_r,i_turb),blade_k_uv(i_phi,i_r,i_turb),&
                            pack(c_uv(i_phi,i_r,i_turb,:),.true.))

                        CTW_origin=compute_vel_3D_cpu(w_,blade_i(i_phi,i_r,i_turb),&
                            blade_j(i_phi,i_r,i_turb),blade_k_w(i_phi,i_r,i_turb),&
                            pack(c_w(i_phi,i_r,i_turb,:),.true.))

                    endif
   
                    !-----------------------------------------------------------
                    ! compute flow
 
                    ! Project from the original grid-align reference to the disk normal reference 
                    CTU1 = CTU1_origin*dcos(yaw_angle(i_turb))+CTV_origin*dsin(yaw_angle(i_turb))
                    CTV = CTV_origin*dcos(yaw_angle(i_turb))-CTU1_origin*dsin(yaw_angle(i_turb))
                    CTW = CTW_origin
                    ! TODO: Check it                    
                    CTV1 = - CTW*cos_Tang(i_phi) + CTV*sin_Tang(i_phi)+omega(i_turb)*(r(i_r,i_turb)*z_i)
                    if (CTV1 > 0) then ! exclude the negative incoming flow
                        CTphi = datan(CTU1/CTV1)
                        sin_CTphi = dsin(CTphi)
                        cos_CTphi = dcos(CTphi)

                        AoA = CTphi*rad2deg - twist(i_r,i_turb)

                        CTUrel  = CTU1/sin_CTphi

                    !    -----------------------------------------------------------
                    !    compute cl/cd from AoA (assume one Re)
                        if (AoA < -90.d0) then
                        print *, 'AoA<-90: me,AoA=',me,AoA,i_r
                        AoA = -90.d0
                        elseif (AoA > 90.d0) then
                        print *, 'AoA<-90: me,AoA=',me,AoA
                        AoA = 90.d0
                        endif

                        call compute_cl_cd_cpu(ctcl,ctcd,AOA,alpha_cl_cd)
                        
                        !-----------------------------------------------------------
                        ! compute BEM

                        ! Tip loss factor
                        f1 = 0.5_fp*3.0_fp*(turb_r(i_turb)-r(i_r,i_turb))/(r(i_r,i_turb)*sin_CTphi)
                        f2 = 2._fp/pi*dacos(dexp(-f1))
                        if (AoA > -10 .and. AoA < 50) then ! Impose a stronger restriction
                        CTFx(i_phi,i_r,i_turb) = 0.5_fp*(CTUrel**2)*CTsoli(i_r,i_turb)*dA(i_r,i_turb)/(dx*dy*dz)*(CTCL*cos_CTphi+CTCD*sin_CTphi)*f2
                        CTFt(i_phi,i_r,i_turb) = 0.5_fp*(CTUrel**2)*CTsoli(i_r,i_turb)*dA(i_r,i_turb)/(dx*dy*dz)*(CTCL*sin_CTphi-CTCD*cos_CTphi)*f2
                        end if
                    end if
                endif
            end do
        end do
    end do
    !---------------------------------------------------------------------------
    ! comunicate

    ! call mpi_allreduce(mpi_in_place, ctfx(1,1,1),&
    !     n_phi*n_r*n_turb,mpi_double_precision,&
    !     mpi_sum,nall,ierr)

    ! call mpi_allreduce(mpi_in_place, ctft(1,1,1),&
    !     n_phi*n_r*n_turb,mpi_double_precision,&
    !     mpi_sum,nall,ierr)

    !---------------------------------------------------------------------------
    ! get thrust and power

    do i_turb=1,n_turb
        do i_r=n_r_s,n_r
            do i_phi=1,n_phi

                thrust(i_turb) = thrust(i_turb)+CTFx(i_phi,i_r,i_turb)*(dx*dy*dz)*z_i**2
                power (i_turb) = power (i_turb)+CTFt(i_phi,i_r,i_turb)*r(i_r,i_turb)*omega(i_turb)*(dx*dy*dz)*z_i**3

            enddo
        enddo
    enddo

    !---------------------------------------------------------------------------
    ! project from BE to cartesian

    do i_turb=1,n_turb
        do i=1,ker_x_width
            do k=2,nzb+1
                do j=int(turb_y(i_turb)/dy)-int(0.5_fp*ker_y_width)+1,int(turb_y(i_turb)/dy)+int(0.5_fp*ker_y_width)
                ! if (flag_j(j)==1) then
                    do i_r=n_r_s,n_r
                        do i_phi=1,n_phi
                            ! if (ker_u(i_phi,i_r,j,k,i,i_turb) >= 1E-15)then
                                ! Rotate the force from the disk-normal reference to the original reference
                                CTFy = CTFt(i_phi,i_r,i_turb)*sin_CTangx(j,k,i,i_turb)
                                ! x_index = turb_i(i_turb)+i-(ker_x_width+1)*0.5
                                ! print *,turb_i(i_turb),turb_x(i_turb)/dx
                                x_index = nint(turb_x(i_turb)/dx)+i-(ker_x_width)*0.5
                                CTFx_origin = CTFx(i_phi,i_r,i_turb)*dcos(yaw_angle(i_turb))-CTFy*dsin(yaw_angle(i_turb))
                                CTFy_origin= CTFx(i_phi,i_r,i_turb)*dsin(yaw_angle(i_turb))+CTFy*dcos(yaw_angle(i_turb))

                                ! Project the force
                                Fx(x_index,j,k) = Fx(x_index,j,k) + CTFx_origin*ker_u(i_phi,i_r,j,k,i,i_turb)
                                Fy(x_index,j,k) = Fy(x_index,j,k) + CTFy_origin*ker_u(i_phi,i_r,j,k,i,i_turb)
                            ! end if

                            ! if (ker_w(i_phi,i_r,j,k,i,i_turb) >= 1E-15)then
                                ! x_index = turb_i(i_turb)+i-(ker_x_width+1)*0.5
                                x_index = nint(turb_x(i_turb)/dx)+i-(ker_x_width)*0.5
                                Fz(x_index,j,k) = Fz(x_index,j,k) + CTFt(i_phi,i_r,i_turb)*ker_w(i_phi,i_r,j,k,i,i_turb)*cos_CTangz(j,k,i,i_turb)
                            ! end if

                        end do
                    end do
                ! endif
                end do
            end do
        end do
    end do


end Subroutine admryaw_cpu

!===============================================================================
! funct lib
!===============================================================================



! compute_cl_cd_cpu
Subroutine compute_cl_cd_cpu(cl,cd,x,alpha_cl_cd)

    implicit none
 

    integer*4 :: t
    real(fp) x,cl,cd
    real(fp), dimension(3,181) :: alpha_cl_cd
    integer :: i,j,k

    ! save alpha_cl_cd

    i=floor(x)+91
    j=ceiling(x)+91

    cl = dble(x-floor(x))*alpha_cl_cd(2,j)+dble(ceiling(x)-x)*alpha_cl_cd(2,i)
    cd = dble(x-floor(x))*alpha_cl_cd(3,j)+dble(ceiling(x)-x)*alpha_cl_cd(3,i)

end Subroutine compute_cl_cd_cpu


!===============================================================================
! fct def for WiRE-01
!===============================================================================
subroutine itp1D_cpu(xData,yData,array_size, xVal,yVal)

    implicit none

    integer, intent(in) :: array_size
    real(fp), intent(in) :: xData(array_size), yData(array_size)
    real(fp),intent(in) :: xVal
    real(fp),intent(out) :: yVal
    integer ::  dataIndex
    real(fp) :: minXdata,maxXdata, xRange, weight

    minXData = xData(1)
    maxXData = xData(size(xData))


    if (xVal<minXData .or. xVal>maxXData) then

        print *,"Error in interpolation"
        stop

    else

        do dataIndex = 1,array_size-1
            if (xVal>xData(dataIndex) .and. xVal<xData(dataIndex+1)) then
                weight = (xVal - xData(dataIndex))/(xData(dataIndex+1)-xData(dataIndex))
                yVal = (1.0-weight)*yData(dataIndex) + weight*yData(dataIndex+1)
                exit
            end if
        end do

    end if

end subroutine itp1D_cpu

! function get_interp_coeff_cpu(blade_ii,blade_jj,blade_kk)
!     implicit none
 

!     real(fp),intent(in) :: blade_ii,blade_jj,blade_kk
!     real(fp),dimension(3) :: blade_point
!     integer*4,dimension(3) :: blade_int
!     integer*4 :: i,j,k,counter
!     real(fp), dimension(3,2) :: l
!     real(fp), dimension(8) :: get_interp_coeff_cpu

!     blade_point = (/blade_ii,blade_jj,blade_kk/)
!     ! Get the location in the cube 
!     do i = 1,3
!         l(i,1) = 1.0 - (dble(blade_point(i)-int(blade_point(i))))
!         l(i,2) = dble(blade_point(i)-int(blade_point(i)))
!     end do
!     ! Get the coefficient
!     counter = 1
!     do i = 1,2
!         do j = 1,2
!             do k = 1,2
!                 get_interp_coeff_cpu(counter) = l(1,i)*l(2,j)*l(3,k)
!                 counter = counter + 1
!             end do
!         end do
!     end do 

! end function get_interp_coeff_cpu

! function compute_vel_3D_cpu(u,blade_i,blade_j,blade_k,c)

!     implicit none
 

!     real(fp), dimension(nx,ny,nz2),intent(in)::u
!     integer*4,intent(in) :: blade_i,blade_j,blade_k
!     integer*4 :: i,j,k,counter
!     real(fp),dimension(8),intent(in) :: c
!     real(fp),dimension(8) :: u_stencil
!     real(fp) :: compute_vel_3D_cpu
!     ! To be rewritten using dot_product
!     counter = 1
!     do i = 0,1
!         do j = 0,1
!             do k = 0,1
!                 u_stencil(counter) = u(blade_i+i,blade_j+j,blade_k+k)
!                 counter = counter + 1            
!             end do
!         end do
!     end do 

!     compute_vel_3D_cpu=dot_product(u_stencil,c)

! end function compute_vel_3D_cpu
!===============================================================================
! fct def for Vestas V80 2MW
!===============================================================================
! Chord length of Vestas V80 2MW (interpolation based on Vestas V90)
! function chord(r)
!
!     implicit none
!     real(fp) :: r,chord
!
!     chord = - 2.8110671517D-8*r**6 + 3.8126820348D-6*r**5 - 2.2363395740D-4*r**4 &
!            + 7.2893308664D-3*r**3 - 1.3510840879D-1*r**2 + 1.1745120720D+0*r**1 &
!            - 9.4466401330D-2
!
! end function chord

! RPM curve of Vestas V80 2MW (interpolation from Hansen et. al 2012)
! function rpm(u_inflow)

!     real(fp) u_inflow,rpm
!     rpm = 4.92921417e-03*u_inflow**5-1.84075843e-01*u_inflow**4+ &
!     2.61464736e+00*u_inflow**3-1.75224421e+01*u_inflow**2+       &
!     5.59968850e+01*u_inflow-5.65494044e+01

! end function rpm

! function rpm_wire01(u_inflow)

!     real(fp) u_inflow,rpm_wire01
!     rpm_wire01 = -11.90213817*u_inflow**3+ 55.83501108*u_inflow**2 + 515.7774735*u_inflow-324.26512599
! end function rpm_wire01



! Pitch angle of Vestas V80 2MW (interpolation based on Vestas V90)
! function pitch(r_,RPM)
!
!     real(fp) r_,r
!     real(fp) Measured_Pitch, RPM, pitch
!     ! TODO: Implement the dynamic change of pitch angle
!     r=r_
!     pitch=-2.0921301914D-4*r**3+3.2105894326D-2*r**2  &
!          -1.5300370009D+0*r**1+2.3276875553D+1
!
! end function pitch
end module turbine_cpu_m
