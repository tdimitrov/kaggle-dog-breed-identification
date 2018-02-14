#!/bin/sh

URL="https://s3.amazonaws.com/kaggle-dogs-data/input.tar.gz"
DEST_DIR="data"

cmd_exists () {
    if [ $# != 1 ]
    then
        echo "Usage: cmd_exists COMMAND"
        exit 1
    fi

    command -v ${1} > /dev/null
    if [ $? != 0 ]
    then
        echo "${1} is not installed"
        exit 1
    fi
}

create_dest () {
    if [ $# != 1 ]
    then
        echo "Usage: create_dest DESTINATION"
        exit 1
    fi

    if [ ! -d ${DEST_DIR} ]
    then
        if [ -f ${DEST_DIR} ]
        then
            echo "${DEST_DIR} exists, but is a file!"
            exit 1
        fi

        mkdir ${DEST_DIR}
    fi
}

cmd_exists "wget"

create_dest ${DEST_DIR}

wget ${URL} -P ${DEST_DIR}/

cd ${DEST_DIR}
tar -zxf input.tar.gz
cd -
