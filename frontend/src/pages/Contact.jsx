import Title from '../components/Title'
import NewsletterBox from '../components/Newsletterbox'
import { assets } from '../assets/assets'


function Contact() {

  return (
    <div>
      <div className='text-center text-2xl pt-10 border-t'>
        <Title text1={'CONTACT'} text2={'US'} />
      </div>
      <div className='my-10 flex flex-col justify-center  md:flex-row gap-10 mb-28'>
        <img className='w-full md:max-w-[480px]' src={assets.contact_img} alt="" />
        <div className='flex flex-col justify-center items-start gap-6'>
          <p className='font-semibold text-xl text-gray-600'>Our Store </p>
          <p className='text-gray-500'>Mainroad, Marthandam</p>
          <p className='text-gray-500'>Tel:9345339017 <br />Email:aadhiaadhi0205@gmail.com</p>
          <p className='font-semibold text-xl text-gray-600 '> Careers at Black and White </p>
          <button className='border border-black px-8 py-4 text-sm hover:bg-black hover:text-white transition-all duration-500 '>Explore Jobs</button>
          <p></p>
          <p></p>
          <p></p>
        </div>
      </div>
      <NewsletterBox />
    </div>
  )
}

export default Contact
