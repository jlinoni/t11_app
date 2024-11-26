import React, { useState } from 'react';

const PatentAnalysisSystem = () => {
  const [currentView, setCurrentView] = useState(1);

  const views = {
    1: {
      title: "Reinvindicaciones y antecedentes",
      leftButton: "Cargar archivo de reinvindicaciones",
      rightButton: "Extraer embeddings >>",
      nextView: 2
    },
    2: {
      title: "Datos de embeddings",
      leftButton: "<< Cargar reinvindicaciones y antecedentes",
      rightButton: "Análisis visual >>",
      prevView: 1,
      nextView: 3
    },
    3: {
      title: "Análisis visual",
      leftButton: "<< Datos de embeddings",
      rightButton: "Búsqueda de novedad de reinvindicacion >>",
      prevView: 2,
      nextView: 4
    },
    4: {
      title: "Resultados de novedad de la reinvindicación",
      leftButton: "<< Análisis visual",
      rightButton: "Cargar otra reinvindicación >>",
      prevView: 3,
      nextView: 1
    }
  };

  const handleNavigation = (direction) => {
    if (direction === 'next') {
      setCurrentView(views[currentView].nextView);
    } else if (direction === 'prev') {
      setCurrentView(views[currentView].prevView);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-gray-100">
      {/* Header */}
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold text-gray-900">
            Sistema de Análisis de Patentes
          </h1>
          <h2 className="text-xl text-gray-600">Visualización y Análisis</h2>
          <p className="text-gray-500">Herramienta para análisis de patentes</p>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-grow container mx-auto px-4 py-8">
        <div className="bg-white shadow rounded-lg p-6">
          {/* View Title */}
          <h2 className="text-2xl font-bold text-center mb-6">
            {views[currentView].title}
          </h2>

          {/* Main Frame */}
          <div className="border-2 border-gray-200 rounded-lg h-96 mb-6">
            {/* Content will go here */}
          </div>

          {/* Navigation Buttons */}
          <div className="flex justify-between mt-4">
            <button
              onClick={() => handleNavigation('prev')}
              className={`px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 
                ${!views[currentView].prevView && 'opacity-50 cursor-not-allowed'}`}
              disabled={!views[currentView].prevView}
            >
              {views[currentView].leftButton}
            </button>
            <button
              onClick={() => handleNavigation('next')}
              className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
            >
              {views[currentView].rightButton}
            </button>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white shadow mt-8">
        <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-gray-50 p-4 rounded">
              <h3 className="font-bold mb-2">Texto de investigación</h3>
              {/* Texto de investigación content */}
            </div>
            <div className="bg-gray-50 p-4 rounded">
              <h3 className="font-bold mb-2">Investigadores</h3>
              <p>Investigador 1 - Investigador 2</p>
            </div>
            <div className="bg-gray-50 p-4 rounded">
              <h3 className="font-bold mb-2">Instituto de Investigación</h3>
              {/* Instituto content */}
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default PatentAnalysisSystem;