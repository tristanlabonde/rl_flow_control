! NOT FINISHED
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

  real(rp), parameter :: max_blow = 0.1_rp ! Maximum blowing force

  integer :: file_unit, io_status, read_status, f_x, f_y
  real(rp) :: f_amp
  
  if (.not. is_initialized) then

    allocate(blow_profile(0:64,0:128))
    blow_profile = 0.0_rp

    file_unit = 99
    open(unit=file_unit, file="wall_blowing_input/single_grid_input.txt", status="old", action="read", iostat=io_status)
    if (io_status /= 0) then
      print *, "ERROR: Unable to open file"
      return
    end if

    do
      ! Lecture des 3 colonnes : X, Y, Amplitude
      read(file_unit, *, iostat=read_status) f_x, f_y, f_amp

      if (read_status < 0) exit
      if (read_status > 0) then
        print *, "ERROR: Error while reading the file"
        exit
      end if

      blow_profile(f_x-100, f_y) = max_blow * f_amp
    end do

    close(file_unit)
    is_initialized = .true.
  end if

  w(101:165,1:129,0) = blow_profile(:,:)
  
  !$acc update device(w(:,:,0))
  !$acc wait
  
end subroutine apply_wall_blowing
end module mod_blowing
