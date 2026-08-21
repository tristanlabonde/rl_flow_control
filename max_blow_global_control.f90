module mod_blowing
contains
subroutine apply_wall_blowing(istep, time, w)
  use mod_param, only: rp
  implicit none

  integer, intent(in) :: istep
  real(rp), intent(in) :: time
  real(rp), intent(inout) :: w(0:,0:,0:)

  real(rp), parameter :: max_blow = 0.8_rp ! Maximum blowing force

  integer :: x, y

  do y = 1, 128
    do x = 101, 164
        w(x, y, 0) = max_blow
    end do
  end do

  !$acc update device(w(:,:,0))

end subroutine apply_wall_blowing
end module mod_blowing
