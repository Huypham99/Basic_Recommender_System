import React from 'react';
import { CardWrapper, ContentWrapper, Title, Brand, Contact } from './style';
import { Col } from 'react-bootstrap'

function SideCard(props) {
    const { type } = props
    return (
        <CardWrapper>
            <Title>
                <h5>{type == 'popular' ? 'Dòng xe phổ biến' : 'Liên hệ'}</h5>
            </Title>

            {
                type == 'popular'
                    ? <ContentWrapper>
                        <Brand>Toyota</Brand>
                        <Brand>Kia</Brand>
                        <Brand>Huyndai</Brand>
                        <Brand>Mazda</Brand>
                        <Brand>Honda</Brand>
                        <Brand>Chervolet</Brand>
                        <Brand>BMW</Brand>
                        <Brand>Ford</Brand>
                    </ContentWrapper>
                    : <ContentWrapper>
                        <Col>
                            <Contact>
                                <p>Quốc Khánh AUTO - SĐT: 090213143</p>
                            </Contact>
                            <p>Địa chỉ - 144, Xuân Thủy, Cầu Giấy</p>
                        </Col>
                    </ContentWrapper>
            }
        </CardWrapper>
    );
}

export default SideCard;