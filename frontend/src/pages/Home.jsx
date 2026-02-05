import { useContext } from 'react'
import Hero from '../components/hero.jsx'
import LatestCollection from '../components/LatestCollection.jsx'
import BestSeller from '../components/BestSeller.jsx'
import OurPolicy from '../components/ourpolicy.jsx'
import NewsletterBox from '../components/Newsletterbox.jsx'

const Home = () => {
  return (
    <div className="text-center py-10">
      <Hero />
      <LatestCollection />
      <BestSeller />
      <OurPolicy />
      <NewsletterBox />
      <h1 className="text-4xl font-bold text-gray-800">Welcome to Black and White</h1>

    </div>
  )
}

export default Home
