import { assets } from './assets/assets.js'
import {Routes,Route} from 'react-router-dom'
import Home from './pages/Home.jsx'
import Collection from './pages/collection.jsx'
import Contact from './pages/Contact.jsx'
import About from './pages/About.jsx'
import Product from './pages/product.jsx'
import Cart from './pages/Cart.jsx'
import Login from './pages/Login.jsx'
import Order from './pages/Order.jsx'
import Placeorder from './pages/Placeorder.jsx'
import Navbar from './components/Navbar.jsx'
import Footer from './components/Footer.jsx'
import Searchbar from './components/Searchbar.jsx'
import { Toaster } from 'react-hot-toast'


const App = () => {
  return (
    <div className='px-4 sm:px-[5vw] md:px-[7vw] lg:px-[9vw]'>
      <Toaster />
      <Navbar/>
      <Searchbar/>
      <Routes>

      <Route path='/' element={<Home/>}  />
      <Route path='/collection' element={<Collection/>} />
      <Route path='/about' element={<About/>} />
      <Route path='/contact' element={<Contact/>} />
      <Route path='/product/:productId' element={<Product/>} />
      <Route path='/cart' element={<Cart/>} />
      <Route path='/Login' element={<Login/>} />
      <Route path='/Order' element={<Order/>} />
      <Route path='/place-order' element={<Placeorder/>} />
            </Routes>
      <Footer/>
    </div>
  )
}

export default App
