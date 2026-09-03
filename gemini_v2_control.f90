module mod_blowing
  use mod_param, only: rp
  implicit none

  logical, save :: is_initialized = .false.
  real(rp), allocatable, save :: blow_profile(:,:)

contains

subroutine apply_wall_blowing(istep, time, w)
  implicit none

  integer, intent(in) :: istep
  real(rp), intent(in) :: time
  real(rp), intent(inout) :: w(0:,0:,0:)

  real(rp), parameter :: max_blow = 0.8_rp

  integer :: file_unit, io_status, read_status, f_x, f_y, gx, gy
  real(rp) :: f_amp
  integer :: nx_max, ny_max, i, j

  nx_max = ubound(w, 1)
  ny_max = ubound(w, 2)

  ! 1. Initialisation unique sur l'hôte (CPU)
  if (.not. is_initialized) then

    allocate(blow_profile(0:nx_max, 0:ny_max))
    blow_profile = 0.0_rp

    file_unit = 99
    open(unit=file_unit, file="wall_blowing_input/single_grid_input.txt", status="old", action="read", iostat=io_status)
    if (io_status == 0) then
      do
        read(file_unit, *, iostat=read_status) f_x, f_y, f_amp
        if (read_status /= 0) exit

        gx = f_x + 1
        gy = f_y + 1
        if (gx <= nx_max .and. gy <= ny_max) then
          blow_profile(gx, gy) = max_blow * f_amp
        end if
      end do
      close(file_unit)
    end if

    ! Copie initiale du profil vers la mémoire GPU
    !$acc enter data copyin(blow_profile)
    is_initialized = .true.
  end if

  ! 2. Application directement DANS LE GPU (évite les transferts PCIe à chaque itération)
  !$acc kernels present(w, blow_profile)
  do j = 0, ny_max
    do i = 0, nx_max
      w(i, j, 0) = blow_profile(i, j)
    end do
  end do
  !$acc end kernels

end subroutine apply_wall_blowing
end module mod_blowing