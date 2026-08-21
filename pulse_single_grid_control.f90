module mod_blowing
contains
subroutine apply_wall_blowing(istep, time, w)
  use mod_param, only: rp
  implicit none

  integer, intent(in) :: istep
  real(rp), intent(in) :: time
  real(rp), intent(inout) :: w(0:,0:,0:)

  real(rp), parameter :: max_blow = 0.8_rp ! Maximum blowing force
  integer, parameter :: pulse_length = 1 ! Length of the blowing pulse

  integer :: file_unit, io_status, read_status, f_x, f_y, gx, gy
  real(rp) :: f_amp

  if (MOD(istep, 10) >= pulse_length) then
    ! blow omly on the first pulse_length time steps of each 10 step
    return
  end if

  f_amp = 0.0_rp
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

    gx = f_x + 1 ! Indice x global
    gy = f_y + 1 ! Indice y global
    w(gx, gy, 0) = max_blow * f_amp
  end do
  
  !$acc update device(w(:,:,0))
  
  close(file_unit)

end subroutine apply_wall_blowing
end module mod_blowing
