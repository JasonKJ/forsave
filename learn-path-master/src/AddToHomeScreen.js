import React, { Component } from "react";

export default class AddToHomeScree extends Component {
  state = {
    enable:
      typeof window.navigator.standalone !== "undefined" &&
      !window.matchMedia("(display-mode: standalone)").matches
  };

  close = () => {
    this.setState({ enable: false });
  };

  componentDidMount() {
    setTimeout(() => {
      this.close();
    }, 10000);
  }

  renderIOSIcon() {
    return (
      <svg
        xmlns="http://www.w3.org/2000/svg"
        width="50"
        height="50"
        viewBox="0 0 50 50"
      >
        <path
          d="M 25 0.59375 L 24.28125 1.28125 L 16.28125 9.28125 A 1.016466
              1.016466 0 1 0 17.71875 10.71875 L 24 4.4375 L 24 32 A 1.0001 1.0001
              0 1 0 26 32 L 26 4.4375 L 32.28125 10.71875 A 1.016466 1.016466 0 1 0
              33.71875 9.28125 L 25.71875 1.28125 L 25 0.59375 z M 7 16 L 7 17 L 7
              49 L 7 50 L 8 50 L 42 50 L 43 50 L 43 49 L 43 17 L 43 16 L 42 16 L 33
              16 A 1.0001 1.0001 0 1 0 33 18 L 41 18 L 41 48 L 9 48 L 9 18 L 17 18 A
              1.0001 1.0001 0 1 0 17 16 L 8 16 L 7 16 z"
          color="#000"
        />
      </svg>
    );
  }

  render() {
    if (!this.state.enable) return "";
    return (
      <div className="addToHome">
        <div>
          To install, click the
          <span className="iOSIcon">{this.renderIOSIcon()}</span> icon below and
          choose 'Add to Home Screen'
        </div>
      </div>
    );
  }
}
