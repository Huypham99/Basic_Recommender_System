import React from 'react';
import {Navbar, Nav, Button} from 'react-bootstrap'
function Navigation(props) {
    return (
        <Navbar bg="dark" variant="dark" expand="lg" sticky="top">
            <Navbar.Brand href="/homepage">Carmazing</Navbar.Brand>
            <Navbar.Toggle aria-controls="basic-navbar-nav" />
            <Navbar.Collapse id="basic-navbar-nav">
                <Nav className="mr-auto">
                    <Nav.Link href="#">Sản Phẩm</Nav.Link>
                    <Nav.Link href="#link">Giới thiệu</Nav.Link>
                    <Nav.Link href="#link">Dịch vụ</Nav.Link>
                    <Nav.Link href="/predict">Tìm kiếm hình ảnh</Nav.Link>
                </Nav>
            </Navbar.Collapse>
            <Button variant="primary" style={{marginRight: '10px'}}>Cần bán xe cũ</Button>
            <Button variant="danger" style={{marginRight: '10px'}}>Nạp xu</Button>
            <Button variant="warning">Đăng tin</Button>
        </Navbar>
    );
}

export default Navigation;