module mod_blowing
contains
subroutine apply_wall_blowing(n, dl, istep, w, lo)
  use mod_param, only: rp
  implicit none

  integer, intent(in) :: n(3), istep, lo(3)
  real(rp), intent(in) :: dl(3)
  real(rp), intent(inout) :: w(0:,0:,0:)

  real(rp), parameter :: max_blow = 5.0_rp ! Maximum blowing force
  real(kind=8), parameter :: pi = 4.0_8 * ATAN(1.0_8)

  integer :: time, x, y, gx, gy

  time = istep - 1
  do y = 1, n(2)
    gy = lo(2) - 1 + y ! Indice y global
    do x = 1, n(1)
      gx = lo(1) - 1 + x ! Indice x global
      if (gx >= 101 .and. gx <= 164 .and. gy >= 1 .and. gy <= 128) then
        w(x, y, 0) = max_blow*sin(2*pi*time/100)
        w(x, y, 1) = max_blow*sin(2*pi*time/100)
      end if
    end do
  end do

  !$acc update device(w(:,:,0:1))

end subroutine apply_wall_blowing
end module mod_blowing
