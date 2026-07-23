module mod_blowing
contains
subroutine apply_wall_blowing(n, dl, istep, w)
  use mod_param, only: rp
  implicit none

  integer, intent(in) :: n(3), istep
  real(rp), intent(in) :: dl(3)
  real(rp), intent(inout) :: w(0:,0:,0:)

  real(rp), parameter :: max_blow = 1.0_rp ! Maximum blowing force

  integer :: time
  integer :: file_unit, io_status, f_x, f_y
  real(rp) :: f_amp

  time = istep - 1
  f_amp = 0.0_rp
  file_unit = 99
  open(unit=file_unit, file="wall_blowing_input/input"//time//".txt", status="old", action="read", iostat=io_status)
  if (io_status /= 0) then
    print *, "ERROR: Unable to open file"
  end if
  do while (io_status == 0)
    ! Lecture des 4 colonnes : Temps, X, Y, Amplitude
    read(file_unit, *) f_x, f_y, f_amp
    w(f_x, f_y, 0) = max_blow * f_amp
  end do
  close(file_unit)

end subroutine apply_wall_blowing
end module mod_blowing
