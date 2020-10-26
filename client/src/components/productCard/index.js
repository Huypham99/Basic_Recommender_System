import React from 'react';
import { ImageWrapper, CardWrapper, ContentWrapper, Price } from './style.js'
import { Button } from 'react-bootstrap'
import { Image } from 'cloudinary-react';

function ProductCard(props) {
    const { name, price, imgId } = props
    return (
        <CardWrapper>
            <Image cloudName="huypham" publicId={imgId} width="328" height='200' crop="scale" />
            <ContentWrapper>
                <h6>{name}</h6>
                <Price>Giá bán: {price} triệu đồng</Price>
                <span>Liên hệ: <b>0906835728</b> </span>
                <Button variant="danger" size="sm" style={{margin: '10px 0'}}>Gọi</Button>
            </ContentWrapper>
        </CardWrapper>
    );
}
export default ProductCard;