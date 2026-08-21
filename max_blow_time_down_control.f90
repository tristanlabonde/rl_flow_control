module mod_blowing
contains
subroutine apply_wall_blowing(istep, time, w)
  use mod_param, only: rp
  implicit none

  integer, intent(in) :: istep
  real(rp), intent(in) :: time
  real(rp), intent(inout) :: w(0:,0:,0:)

  real(rp), parameter :: max_blow = 0.9_rp ! Maximum blowing force
  integer :: freq = 1

  integer :: x, y

  do y = 1, 128
    do x = 101, 164
        w(x, y, 0) = max_blow - (istep/freq)*0.1
    end do
  end do

  ! if (mod(istep, 1000) == 0) then
  !   print *, "istep", istep, "w", max_blow - (istep/1000)*0.1
  ! endif

  !$acc update device(w(:,:,0))

end subroutine apply_wall_blowing
end module mod_blowing
