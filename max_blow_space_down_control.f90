module mod_blowing
contains
subroutine apply_wall_blowing(n, istep, time, w)
  use mod_param, only: rp
  implicit none

  integer, intent(in) :: n(3), istep
  real(rp), intent(in) :: time
  real(rp), intent(inout) :: w(0:,0:,0:)

  real(rp), parameter :: max_blow = 0.8_rp ! Maximum blowing force
  integer :: rate = 5

  integer :: x, y

  do y = 1, n(2)
    do x = 101, 164
      if (mod(((x-101)/4)*rate, 32) < 16) then
        w(x, y, 1) = max_blow - mod(((x-101)/4)*rate, 16)*0.1
      else
        w(x, y, 1) = - max_blow + mod(((x-101)/4)*rate, 16)*0.1
      end if
    end do
  end do

  ! if (mod(istep, 1000) == 0) then
  !   print *, "istep", istep, "w", max_blow - (istep/1000)*0.1
  ! endif

  !$acc update device(w(:,:,1))

end subroutine apply_wall_blowing
end module mod_blowing
