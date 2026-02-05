import { useContext } from 'react';
import { assets } from '../assets/assets';

const Footer = () => {
  return (
    <div className="mt-40 text-sm text-gray-600">

      {/* Top Footer Section */}
      <div className="flex flex-col sm:grid sm:grid-cols-[3fr_1fr] gap-14 my-10">

        {/* Left — Logo & Description */}
        <div>
          <div className="flex flex-col items-start mb-5">
            <img src={assets.logo} className="w-32" alt="Logo" />
            <p className='text-sm font-semibold tracking-wider uppercase text-gray-800 mt-1'>BLACK AND WHITE</p>
          </div>
          <p className="w-full md:w-2/3">
            Lorem ipsum is simply dummy text used to fill space in a layout.
          </p>
        </div>

        {/* Right — Contact Info */}
        <div>
          <h3 className="text-xl font-medium mb-5 text-gray-800">GET IN TOUCH</h3>
          <ul className="flex flex-col gap-2">
            <li>9345339017</li>
            <li>contact@blackandwhite.com</li>
          </ul>
        </div>
      </div>

      {/* Bottom Copyright */}
      <hr className="border-gray-300" />
      <p className="py-5 text-sm text-center text-gray-500">
        © 2025 blackandwhite.com — All rights reserved.
      </p>
    </div>
  );
};

export default Footer;

