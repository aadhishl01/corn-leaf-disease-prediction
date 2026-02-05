import { useState, useContext } from 'react'
import { assets } from '../assets/assets'
import { Link, NavLink } from 'react-router-dom'
import { ShopContext } from '../context/shopcontext'

const Navbar = () => {
  const [visible, setVisible] = useState(false); // For mobile sidebar
  const [dropdown, setDropdown] = useState(false); // For profile dropdown
  const { setshowsearch, getCartCount, user, logout } = useContext(ShopContext);

  return (
    <div className='flex items-center justify-between py-5 font-medium'>
      {/* Logo */}
      <Link to='/' className='flex flex-col items-center'>
        <img src={assets.logo} className='w-36' alt="Logo" />
        <p className='text-sm font-semibold tracking-wider uppercase text-gray-800 mt-1'>BLACK AND WHITE</p>
      </Link>

      {/* Navlinks */}
      <ul className='flex gap-12 text-sm text-gray-700'>
        <NavLink to='/' className='flex flex-col items-center gap-1 mx-2'>
          <p>HOME</p>
          <hr className='w-2/4 border-none h-[1.5px] bg-gray-700 hidden' />
        </NavLink>

        <NavLink to='/collection' className='flex flex-col items-center gap-1 mx-2'>
          <p>COLLECTION</p>
          <hr className='w-2/4 border-none h-[1.5px] bg-gray-700 hidden' />
        </NavLink>

        <NavLink to='/about' className='flex flex-col items-center gap-1 mx-2'>
          <p>ABOUT</p>
          <hr className='w-2/4 border-none h-[1.5px] bg-gray-700 hidden' />
        </NavLink>

        <NavLink to='/contact' className='flex flex-col items-center gap-1 mx-2'>
          <p>CONTACT</p>
          <hr className='w-2/4 border-none h-[1.5px] bg-gray-700 hidden' />
        </NavLink>
      </ul>

      {/* Icons Section */}
      <div className='flex items-center gap-6'>
        {/* Search */}
        <Link to='/login'> <img onClick={() => setshowsearch(true)} src={assets.search_icon} className='w-5 cursor-pointer' alt="Search" /></Link>

        {/* Profile Dropdown */}
        <div className='group relative'>
          {
            user ? (
              <>
                <img
                  src={assets.profile_icon}
                  className='w-5 cursor-pointer'
                  alt="Profile"
                />
                <div className='group-hover:block hidden absolute right-0 pt-4'>
                  <div className='flex flex-col gap-2 w-36 py-3 px-5 bg-slate-100 text-gray-500 rounded shadow-md'>
                    <p className='cursor-pointer hover:text-black'>My Profile</p>
                    <Link to='/Order' className='cursor-pointer hover:text-black'>Orders</Link>
                    <p onClick={logout} className='cursor-pointer hover:text-black'>Logout</p>
                  </div>
                </div>
              </>
            ) : (
              <Link to='/login'>
                <img src={assets.profile_icon} className='w-5 cursor-pointer' alt="Login" />
              </Link>
            )
          }
        </div>

        {/* Cart */}
        <NavLink to='/cart' className='relative'>
          <img src={assets.cart_icon} className='w-5 min-w-5' alt="Cart" />
          <p className='absolute right-[-5px] bottom-[-5px] w-4 text-center leading-4 bg-black text-white aspect-square rounded-full text-[8px]'>{getCartCount()}</p>
        </NavLink>

        {/* Menu Icon (Mobile) */}
        <img
          onClick={() => setVisible(true)}
          src={assets.menu_icon}
          className='w-6 cursor-pointer sm:hidden'
          alt="Menu"
        />
      </div>

      {/* Mobile Sidebar Menu */}
      <div className={`absolute top-0 right-0 bottom-0 overflow-hidden bg-white transition-all duration-300 ${visible ? 'w-full' : 'w-0'}`}>
        <div className='flex flex-col text-gray-600'>
          {/* Close Button */}
          <div onClick={() => setVisible(false)} className='flex items-center justify-end p-3 cursor-pointer'>
            <img src={assets.cross_icon} className='h-4 rotate-180' alt="Close" />
          </div>
          {/* Sidebar Nav Links */}
          <NavLink onClick={() => setVisible(false)} className='py-2 pl-6 border' to='/'>
            HOME
          </NavLink>
          <NavLink onClick={() => setVisible(false)} className='py-2 pl-6 border' to='/collection'>
            COLLECTION
          </NavLink>
          <NavLink onClick={() => setVisible(false)} className='py-2 pl-6 border' to='/about'>
            ABOUT
          </NavLink>
          <NavLink onClick={() => setVisible(false)} className='py-2 pl-6 border' to='/contact'>
            CONTACT
          </NavLink>
        </div>
      </div>
    </div>
  )
}

export default Navbar
