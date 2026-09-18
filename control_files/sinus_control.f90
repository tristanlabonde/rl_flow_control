! Flow control file used as control.f90 in the CaNS code.
! Reads data file containing blowing scheme - w velocities (z axis) - to apply as flow control in CaNS simulation.
! This code applies a blowing scheme based on the computation of the sinus fonction multiplicated by max_blow.
module mod_blowing
contains
subroutine apply_wall_blowing(istep, time, w)
  use mod_param, only: rp
  implicit none

  integer, intent(in) :: istep
  real(rp), intent(in) :: time
  real(rp), intent(inout) :: w(0:,0:,0:)

  real(rp), parameter :: max_blow = 1.0_rp ! Maximum blowing force
  real(kind=8), parameter :: pi = 4.0_8 * ATAN(1.0_8)

  integer :: x, y

  integer :: io_status
  integer, parameter :: file_unit = 99

  open(unit=file_unit, file="sinus_control_values_debug.txt", status="old", action="replace", iostat=io_status)
  if (io_status /= 0) then
    print *, "ERROR: Unable to open file"
    return
  end if
  write(file_unit, *) "x y w(x, y, 0)"

  do y = 1, 128
    do x = 101, 164
      w(x, y, 0) = max_blow*sin(2*pi*time/100)
      write(file_unit, *) x, y, w(x, y, 0)
    end do
  end do

  !$acc update device(w(:,:,0))

end subroutine apply_wall_blowing
end module mod_blowing
