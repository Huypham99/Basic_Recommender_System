import React, { useEffect, useState } from 'react';
import { Button, Container, Row, Col } from 'react-bootstrap'
import car1 from '../../images/carouselImages/car1.jpg'
import ProductCard from '../../components/productCard'
import axios from 'axios'
import './style.css'

function Prediction(props) {

    const [file, setFile] = useState('')
    const [imageURL, setImageURL] = useState('')
    const [prediction, setPrediction] = useState('')
    const [display, setDisplay] = useState('')
    const [btnDisplay, setBtnDisplay] = useState('none')
    const [loader, setLoader] = useState('none')
    const [car, setCar] = useState([])

    const uploadImage = (e) => {

        e.preventDefault()

        let formData = new FormData();

        formData.append('file', file);

        setLoader('block')

        axios.post(`/predict`, formData)
            .then(res => {
                //setPrediction(res.data)
                setLoader('none')
                setDisplay('block')
                setCar(res.data)
            })
    }

    const chooseFile = (e) => {
        const file = e.target.files[0]
        setFile(file)
        setImageURL(URL.createObjectURL(file))
        setDisplay('none')
        setCar('')
        setBtnDisplay('block')
    }

    const myStyle = {
        borderRight: 'solid 2px lightgrey',
        padding: "10px"
    };

    return (
        <Container>
            <Row>
                <Col lg="6" style={myStyle}>
                    <h4>Tải ảnh lên:</h4>
                    <form encType="multipart/form-data" onSubmit={uploadImage}>
                        <input type='file' onChange={chooseFile} placeholder="Choosing image" />
                    </form>
                    <img src={imageURL} style={{ marginTop: '10px' }} width='400' /><br></br>
                    <Button variant="primary" onClick={uploadImage} disabled={!file} style={{ marginTop: '10px', display: btnDisplay }}>Tra cứu</Button>
                </Col>
                <Col lg="6">
                    <h4>Sản phẩm có thể bạn quan tâm:</h4>
                    {car ? car.map(x => <Col className='fade-in' style={{ display: display }}><ProductCard name={x.name} price={x.price} imgId={x.imageId} /></Col>) : <div class="loader" style={{ display: loader }}></div>}
                </Col>
            </Row>
        </Container>
    )
}

export default Prediction;