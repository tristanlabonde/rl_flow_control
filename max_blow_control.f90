module mod_blowing
contains
subroutine apply_wall_blowing(n, dl, istep, w)
  use mod_param, only: rp
  implicit none

  integer, intent(in) :: n(3), istep
  real(rp), intent(in) :: dl(3)
  real(rp), intent(inout) :: w(0:,0:,0:)

  real(rp), parameter :: max_blow = 2.0_rp ! Maximum blowing force

  integer :: time, x, y

  do x = 100, 164, 1
    do y = 0, 128, 1
      w(x, y, 0) = max_blow
    end do
  end do

  !$acc update device(w(:,:,0))

end subroutine apply_wall_blowing
end module mod_blowing
