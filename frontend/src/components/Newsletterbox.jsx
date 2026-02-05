const NewsletterBox = () => {
const onSubmitHandler =(event) =>{
    event.preventDefault();
}



  return (
    <div className='text-center'>
        <p className='text-2xl font-medium text-gray-800'>
        Subscribe Now to get 20% offer!
        </p>
        <p className='text-gray-400 mt-3'>
          Welcome to the Black and White shopping website
        </p>
        <form onSubmit={onSubmitHandler} className='w-full sm:w-1/2 flex items-center gap-3 mx-auto my-6 border pl-3'>
          <input className='w-full sm:flex-1 outline-none' type="email" placeholder='Enter your email' required />
        <button type='submit' className='bg-black text-white text-xs px-10 py-4'>SUBSCRIPTION</button>
        </form>

         </div>
  )
}

export default NewsletterBox
