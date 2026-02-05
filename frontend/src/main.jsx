import React, { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import { BrowserRouter } from 'react-router-dom'
import ShopContextProvider from './context/shopcontext.jsx'
import ErrorBoundary from './components/ErrorBoundary.jsx'


const rootElement = document.getElementById('root')

if (rootElement) {
  createRoot(rootElement).render(
    <StrictMode>
      <ErrorBoundary>
        <BrowserRouter>
          <ShopContextProvider>
            <App />
          </ShopContextProvider>
        </BrowserRouter>
      </ErrorBoundary>
    </StrictMode>
  )
} else {
  console.error("Root element not found!")
}
