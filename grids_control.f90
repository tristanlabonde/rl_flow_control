module mod_blowing
contains
subroutine apply_wall_blowing(istep, time, w)
  use mod_param, only: rp
  implicit none

  integer, intent(in) :: istep
  real(rp), intent(in) :: time
  real(rp), intent(inout) :: w(0:,0:,0:)

  real(rp), parameter :: max_blow = 0.8_rp ! Maximum blowing force

  integer :: file_unit, io_status, read_status, f_x, f_y, gx, gy
  real(rp) :: f_amp

  character(len=10) :: step_str

  write(step_str, '(I0)') (istep - 1)
  f_amp = 0.0_rp
  file_unit = 99
  open(unit=file_unit, file="wall_blowing_input/grids_input"//trim(adjustl(step_str))//".txt", status="old", action="read", iostat=io_status)
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
