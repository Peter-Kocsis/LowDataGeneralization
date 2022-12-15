#!/bin/bash

./waved &

pushd ..
wave run gui.adl4cv_app.adl4cv_app
popd

pkill -P $$
