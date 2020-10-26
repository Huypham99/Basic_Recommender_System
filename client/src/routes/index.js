import React from 'react';
import { BrowserRouter as Router, Switch, Route, Redirect } from 'react-router-dom'
import Homepage from '../views/Homepage'
import Prediction from '../views/Prediction'
import Navigation from '../views/Navigation';

const Routes = () => {

    return (
        <Router>
            <Route component={Navigation}/>
            <Switch>
                    <Route path='/homepage' exact component={Homepage} />
                    <Route path='/predict' component={Prediction} />
            </Switch>
        </Router>
    );
};

export default Routes;