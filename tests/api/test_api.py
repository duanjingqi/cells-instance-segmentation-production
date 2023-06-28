# Configure Python environment
import random
from typing import List, Union
from pathlib import Path, PosixPath

from starlette.testclient import TestClient
from starlette.status import HTTP_200_OK


def generate_multipartformdata(input: Union[Path, List[Path]]) -> List:
    """Generate multipart/form-data from input list"""
    files = []

    if type(input) == PosixPath:
        filename = input.name
        mime = f'image/{input.suffix[1:]}'
        files.append((filename, input.open(), mime))

    else:
        
        for fp in input: 
            filename = fp.name
            mime = f'image/{fp.suffix[1:]}'
            files.append((filename, fp.open(), mime))

    return files

# Images for test

def test_single_prediction(image_path, test_client: TestClient):

    selected_single = random.choice(list(image_path.glob('*')))
    data = generate_multipartformdata(selected_single)
    response = test_client.post('/predict/single', files = {'infile': data})

    assert response.status_code == HTTP_200_OK
    assert response[0]['Prediction'].shape == (256, 256, 3)
 

def test_batch_prediction(image_path, test_client: TestClient):

    all_images = list(image_path.glob('*'))
    data = generate_multipartformdata(all_images)
    response = test_client.post('/predict/batch', files = {'infiles': data})

    assert response.status_code == HTTP_200_OK
    assert len(response) == len(data)
    assert response[random.randint(0, len(data))]['Prediction'].shape == (256, 256, 3)