! Flow control file used as control.f90 in the CaNS code.
! This code applies a sinusoidal-stairs-like control in space (x axis)
! e.g. blowing for the 4 first x coordinates at max_blow then, for the next 4 x coordinate, blow at max_blow - 0.1*rate, etc.
! Blow stays between max_blow and - maxblow.
module mod_blowing
contains
subroutine apply_wall_blowing(istep, time, w)
  use mod_param, only: rp
  implicit none

  integer, intent(in) :: istep
  real(rp), intent(in) :: time
  real(rp), intent(inout) :: w(0:,0:,0:)

  real(rp), parameter :: max_blow = 0.8_rp ! Maximum blowing force
  integer :: rate = 5

  integer :: x, y

  do y = 1, 128
    do x = 101, 164
      if (mod(((x-101)/4)*rate, 32) < 16) then
        w(x, y, 0) = max_blow - mod(((x-101)/4)*rate, 16)*0.1
      else
        w(x, y, 0) = - max_blow + mod(((x-101)/4)*rate, 16)*0.1
      end if
    end do
  end do

  ! if (mod(istep, 1000) == 0) then
  !   print *, "istep", istep, "w", max_blow - (istep/1000)*0.1
  ! endif

  !$acc update device(w(:,:,0))

end subroutine apply_wall_blowing
end module mod_blowing
