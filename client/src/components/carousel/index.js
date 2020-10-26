import React, { useState } from 'react';
import car1 from '../../images/carouselImages/car1.jpg'
import car2 from '../../images/carouselImages/car2.jpg'
import car3 from '../../images/carouselImages/car3.jpg'
import { Carousel } from 'react-bootstrap'

function ControlledCarousel() {
    const [index, setIndex] = useState(0);

    const handleSelect = (selectedIndex, e) => {
        setIndex(selectedIndex);
    };

    return (
        <Carousel activeIndex={index} onSelect={handleSelect} style={{margin: '20px 0'}}>
            <Carousel.Item>
                <img
                    src={car1}
                    alt="First slide"
                />
            </Carousel.Item>
            <Carousel.Item>
                <img
                    src={car2}
                    alt="Second slide"
                />
            </Carousel.Item>
            <Carousel.Item>
                <img
                    src={car3}
                    alt="Third slide"
                />
            </Carousel.Item>
        </Carousel>
    );
}

export default ControlledCarousel;