module mod_blowing
contains
subroutine apply_wall_blowing(n, istep, time, w)
  use mod_param, only: rp
  implicit none

  integer, intent(in) :: n(3), istep
  real(rp), intent(in) :: time
  real(rp), intent(inout) :: w(0:,0:,0:)

  real(rp), parameter :: max_blow = 0.8_rp ! Maximum blowing force

  integer :: x, y

  do y = 1, n(2)
    do x = 101, 164
        w(x, y, 1) = max_blow
    end do
  end do

  !$acc update device(w(:,:,1))

end subroutine apply_wall_blowing
end module mod_blowing
