module fft_mod
    implicit none
    integer,       parameter :: dp=selected_real_kind(15,300)
    real(kind=dp), parameter :: pi=3.141592653589793238460_dp
contains
   
    ! In place Cooley-Tukey FFT
    recursive subroutine fft(y)
      complex(kind=dp), dimension(:), intent(inout)  :: y
      complex(kind=dp)                               :: t
      integer                                        :: N
      integer                                        :: i
      complex(kind=dp), dimension(:), allocatable    :: even, odd
   
      N=size(y)
   
      if(N .le. 1) return
   
      allocate(odd((N+1)/2))
      allocate(even(N/2))
   
      ! divide
      odd =y(1:N:2)
      even=y(2:N:2)
   
      ! conquer
      call fft(odd)
      call fft(even)
   
      ! combine
      do i=1,N/2
         t=exp(cmplx(0.0_dp,-2.0_dp*pi*real(i-1,dp)/real(N,dp),kind=dp))*even(i)
         y(i)     = odd(i) + t
         y(i+N/2) = odd(i) - t
      end do
   
      deallocate(odd)
      deallocate(even)
   
    end subroutine fft
   
end module fft_mod
   
program test
    use fft_mod
    use omp_lib

    implicit none
    complex(kind=dp), dimension(:, :), allocatable :: data 
    integer :: i, j, n, m
    real :: start, finish
    n = 1024
    m = 8192
    allocate(data(m, n))
    do i=1, n/2 
        do j=1, m
            data(j, i) = 1
        end do
    end do


    !$acc data copy(data)
    call cpu_time(start)

    !$acc parallel loop   
    do i=1,m     
        call fft(data(i, :))
    end do
    !$acc end parallel loop

    call cpu_time(finish)

    !$acc end data
    print '("Time = ",f6.4," seconds.")', finish-start


    ! do i=1,n
    !     write(*,'("(", F20.15, ",", F20.15, "i )")') data(2, i)
    ! end do
   
    ! write(*, *) data(1, :)
end program test