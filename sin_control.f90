module mod_blowing
contains
subroutine apply_wall_blowing(n, dl, istep, w)
  use mod_param, only: rp
  implicit none

  integer, intent(in) :: n(3), istep
  real(rp), intent(in) :: dl(3)
  real(rp), intent(inout) :: w(0:,0:,0:)

  real(rp), parameter :: max_blow = 1.0_rp ! Maximum blowing force
  real(kind=8), parameter :: pi = 4.0_8 * ATAN(1.0_8)

  integer :: time, x, y

  do x = 100, 164, 1
    do y = 0, 128, 1
      w(x, y, 0) = max_blow*sin(2*pi*time/100)
    end do
  end do

end subroutine apply_wall_blowing
end module mod_blowing
