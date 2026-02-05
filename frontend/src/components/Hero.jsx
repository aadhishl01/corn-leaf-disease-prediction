import { useContext } from 'react'
import { assets } from '../assets/assets'
import { NavLink } from 'react-router-dom'

const Hero = () => {
  return (
    <div className='flex flex-col sm:flex-row border border-gray-400'>
      {/* Hero left side */}
      <div className='w-full sm:w-1/2 flex items-center justify-center py-10 sm:py-0 px-6 sm:px-12'>
        <div className='text-[#414141] text-center sm:text-left'>
          <div className='flex items-center justify-center sm:justify-start gap-2'>
            <p className='w-8 md:w-11 h-[2px] bg-[#414141]'></p>
            <p className='font-medium text-sm md:text-base tracking-wide'>OUR BESTSELLER</p>
          </div>

          <h1 className='text-3xl sm:py-3 lg:text-5xl font-bold leading-relaxed'>
            Latest Arrivals
          </h1>

          <div className='flex items-center justify-center sm:justify-start gap-3 mt-2'>
            <NavLink
              to='/collections'
              className='bg-[#414141] text-white text-sm md:text-base font-medium py-2 px-4 rounded-md hover:bg-[#2c2c2c] transition-all duration-200'
            >
              SHOP NOW
            </NavLink>
            <p className='w-8 md:w-11 h-[2px] bg-[#414141]'></p>
          </div>
        </div>
      </div>

      {/* Hero right side */}
      <img
        src={assets.hero_img}
        className='w-full sm:w-1/2 object-cover'
        alt='Latest fashion arrivals'
      />
    </div>
  )
}

export default Hero
