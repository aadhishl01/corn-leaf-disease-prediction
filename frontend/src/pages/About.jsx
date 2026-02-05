import { useContext } from 'react'
import Title from '../components/Title'
import Newsletterbox from '../components/Newsletterbox'
import { assets } from '../assets/assets'

const About = () => {
  return (
    <div>
    <div className='text-2xl text-center pt-8 border-t'>
      <Title text1={'ABOUT'} text2={'US'} />
    </div>
    <div className='my-10 flex flex-col md:flex-row gap-16'>
      <img  className ='w-full  md:max-w-[450px]'src={assets.about_img} alt="" />
      <div className='flex flex-col justify-center gap-6 md:w-2/4 text-gray-600'>
      <p>Lorem ipsum dolor sit amet consectetur adipisicing elit. Consectetur iure optio facilis, a repellendus eos inventore voluptates odit nesciunt eaque blanditiis aliquid tenetur debitis obcaecati omnis laboriosam dicta delectus recusandae!</p>
      <p>hello welcome to our shopping website hope you have a nice day</p>
      <b>Our Mission</b>
      <p> Page layouts look better with something in each section. Web page designers, content writers, and layout artists use lorem ipsum, also known as placeholder copy, to distinguish which areas on a page will hold advertisements, editorials</p>
      </div>
    </div>
    <div className='text-xl py-4'>
      <Title text1={'WHY'} text2={'CHOOSE US'} />
    </div>
    <div className='flex flex-col md:flex-row text-sm mb-20'>
      <div className='border px-10 md:px-16 py-8 sm:py-20 flex flex-col gap-5'>
        <b>QUALITY ASSURANCE:</b>
        <p className='text-gray-600'>Lorem ipsum dolor sit amet consectetur adipisicing elit. Recusandae iusto rerum doloribus nisi soluta, deserunt esse veritatis corporis reiciendis, ad ab! Modi expedita iure deleniti saepe facere dicta error tempora.</p>
      </div>
      <div className='border px-10 md:px-16 py-8 sm:py-20 flex flex-col gap-5'>
        <b>Convience:</b>
        <p className='text-gray-600'>With our friendly wishes happy stay at the store visit again</p>
      </div>
       <div className='border px-10 md:px-16 py-8 sm:py-20 flex flex-col gap-5'>
        <b>Exceptional Customer Service</b>
        <p className='text-gray-600'>It's deliberately scrambled Latin that doesn't form coherent sentences. While it comes from Cicero's "De Finibus Bonorum et Malorum," the text has been modified so extensively that it's nonsensical</p>
      </div>
    </div>
    <Newsletterbox/>
    </div>
  )
}

export default About
