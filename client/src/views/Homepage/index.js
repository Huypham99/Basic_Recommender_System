import React, { useEffect, useState } from 'react';
import { Card, Button, Row, Col, Container } from 'react-bootstrap'
import car1 from '../../images/carouselImages/car1.jpg'
import Carousel from '../../components/carousel'
import ProductCard from '../../components/productCard'
import SideCard from '../../components/sideCard'
import axios from 'axios'
import './index.css'

function Homepage(props) {

    const [cars, setCars] = useState()

    useEffect(() => {
        axios.get(`/cars`)
            .then(res => {
                setCars(res.data)
            })
    }, []);

    return (
        <Container className='container' fluid>

            <Col>
                <Col><SideCard type='popular' /></Col>
                <Col><SideCard type='contact' /></Col>
            </Col>
            <Col>
                <Row className="justify-content-md-center">
                    <Carousel />
                </Row>
                <Row className="justify-content-md-center">
                    <h5>Tất cả sản phẩm</h5>
                </Row>
                <Row className="justify-content-md-center" className="product_container">
                    {cars && cars.map(car => <Col><ProductCard name={car.name} price={car.price} imgId={car.imageId} /></Col>)}
                </Row>
            </Col>
        </Container>
    );
}

export default Homepage;