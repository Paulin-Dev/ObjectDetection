import fs from 'fs';
import { readFile, readdir } from 'fs/promises';
import { Readable } from 'stream';


async function readJsonFile(filename) {
    try {
        const data = await readFile(filename, 'utf8');
        return JSON.parse(data);
    } catch (err) {
        console.error(err);
    }
}


async function downloadImage(filename, project_id, path) {
    const url = `https://app.heartex.com/storage-data/uploaded/?filepath=upload/${project_id}/${filename}`;
    try {
        const response = await fetch(url, { headers: { Authorization: `Token ${process.env.ACCESS_TOKEN}` }});
        if (response.ok && response.body) {
            const fileStream = fs.createWriteStream(`${path}/${filename}`);
            Readable.fromWeb(response.body).pipe(fileStream);
        } else {
            console.error(response.status, response.statusText)
        }
    } catch (err) {
        console.error(err);
    }
   
}


switch (process.env.EXPORTED_FORMAT.toLowerCase()) {

    case 'json':
        json();
        break;
    
    case 'coco':
        coco();
        break;

    case 'yolo':
        yolo();
        break;

    default:
        break;

}


async function json() {
    const data = await readJsonFile('exported');
    data.forEach((item) => {
        downloadImage(item.file_upload, item.project, '/json_images');
    });
}


async function coco() {
    const data = await readJsonFile('exported/result.json');
    data?.images.forEach((image) => {
        const filename = image.file_name.substring(image.file_name.indexOf('/') + 1);
        downloadImage(filename, process.env.PROJECT_ID, 'exported/images');
    });
}


async function yolo() {
    const data = await readdir('exported/labels'); 
    data.forEach((item) => {
        const index = item.lastIndexOf('.');
        if(item.substring(index) == '.txt') {
            downloadImage(item.substring(0, index) + '.jpg', process.env.PROJECT_ID, 'exported/images');
        }
    });
}
