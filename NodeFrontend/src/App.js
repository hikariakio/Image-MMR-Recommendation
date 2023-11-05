import logo from "./logo.svg";
import "./App.css";
import "./style.css";

import { useEffect, useState } from "react";
import axios from "axios";
import { PairButtons } from "./components/PairButtons";
import Slider from "@mui/joy/Slider";
import Grid from '@mui/joy/Grid';
import LoadingScreen from "./components/LoadingScreen";
import Header from "./Header";


function App() {
  const [data, setData] = useState(null);
  const [sliderValue, setSliderValue] = useState(0.5); // Default value of slider

  const [mode, setMode] = useState("Maximum");
  const [model, setModel] = useState("GoogleNews-word2vec");

  const [loading,setLoading] = useState(false);
  useEffect(() => {
      const fetchData = async () => {
          setLoading(true);

          try {
        const response = await axios.get(
          "http://localhost:5000/api/random-images",
        );

        setData(response.data);
          setLoading(false);
      } catch (error) {
        console.error("Error fetching data: ", error);
      }
    };

    fetchData();
  }, []);

  const handleImageClick = async (id) => {
    try {
        setLoading(true);

        const response = await axios.post(
        "http://localhost:5000/api/log-image-click",
        {
          id, // Send the id in the request body
          lambda_value: sliderValue, // Passing slider value as lambda_value
          mode: mode, //max
          model: model, //glove
        },
      );
      setData(response.data);
        setLoading(false);

      console.log(response.data.message); // Log the success message
    } catch (error) {
      console.error("Error logging image click:", error);
    }
  };

  return (
      <>
      <Header/>
    <div>
        <Grid container>
        <Grid sm={12} lg={6}>
          <PairButtons
            id={"Vector Model"}
            button1Name={"GoogleNews-word2vec"}
            button2Name={"Wiki-gigaword-glove"}
            selectedButton={model}
            setSelectedButton={setModel}
          />
          <PairButtons
            id={"MMR Mode"}
            button1Name={"Maximum"}
            button2Name={"Average"}
            selectedButton={mode}
            setSelectedButton={setMode}
          />
        </Grid>
            <Grid xs={4} className={"slider-container"}>
                <h3> Lambda Value (0-1)</h3>
                <p> 0 for diversity , 1 for maximum similarity</p>
          <Slider
            defaultValue={0.5}

            min={0}
            max={1}
            step={0.01}
            valueLabelDisplay="auto"
            onChange={(x) => {
              setSliderValue(x.target.value);
            }}
            size="md"
          />
        </Grid>
      </Grid>
      <div className="grid">
        {data
          ? data.map((id, index) => (
              <img
                key={index}
                src={`http://localhost:5001/${id
                  .toString()
                  .padStart(12, "0")}.jpg`}
                alt={`Image ${id}`}
                className="grid-item"
                onClick={() => handleImageClick(id)} // Add onClick event handler
              />
            ))
          : "Loading..."}
      </div>
        {loading&&<LoadingScreen loadingtext={"Loading"}/>}
    </div>
      </>
  );
}

export default App;
