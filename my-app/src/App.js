import React from 'react';
import logo from './huji.png';
import './App.css';
// Font Awesomeのインポート
import { FaCog } from 'react-icons/fa';


function App() {
  return (
    <div className="App">
      <header className="App-header">
        <a>
          ふじいくん
        </a>
        <img src={logo} className="App-logo" alt="logo" />
        <button>
          対局開始
        </button>
        <button>
          対局履歴
        </button>
        <div className="settings-button">
          <FaCog size={24} />
        </div>
      </header>
    </div>
  );
}

export default App;
