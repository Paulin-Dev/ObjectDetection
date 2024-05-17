import fs from 'fs';
import { readFile, readdir } from 'fs/promises';
import { Readable } from 'stream';


const BASE_URL = 'https://app.heartex.com/storage-data/uploaded/?filepath=upload';


// Read JSON file and return the parsed data
async function readJsonFile(filename) {
    try {
        const data = await readFile(filename, 'utf8');
        return JSON.parse(data);
    } catch (err) {
        console.error(err);
    }
}


// Check if the file exists
function fileExists(path) {
    try {
        return fs.existsSync(path);
    } catch (err) {
        return false;
    }
}


// Download image from the given URL
async function downloadImage(filename, project_id, path) {
    if (fileExists(`${path}/${filename}`)) { return; }

    const url = `${BASE_URL}/${project_id}/${filename}`;
    try {
        const response = await fetch(url, { headers: { Authorization: `Token ${process.env.ACCESS_TOKEN}` }});
        if (response.ok && response.body) {
            const fileStream = fs.createWriteStream(`${path}/${filename}`);
            Readable.fromWeb(response.body).pipe(fileStream);
        } else {
            console.error(response.status, response.statusText);
        }
    } catch (err) {
        console.error(err);
    }
   
}


// Exported format
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


// Download images from the JSON file
async function json() {
    const data = await readJsonFile('exported');
    let counter = 0;
    data.forEach((item) => {
        setTimeout(downloadImage, counter, item.file_upload, item.project, '/json_images')
        counter += parseInt(process.env.REQUEST_INTERVAL);
    });
}


// Download images from the COCO file
async function coco() {
    const data = await readJsonFile('exported/result.json');
    let counter = 0;
    data?.images.forEach((image) => {
        const filename = image.file_name.substring(image.file_name.indexOf('/') + 1);
        setTimeout(downloadImage, counter, filename, process.env.PROJECT_ID, 'exported/images');
        counter += parseInt(process.env.REQUEST_INTERVAL);
    });
}


// Download images from the YOLO file
async function yolo() {
    const data = await readdir('exported/labels');
    let counter = 0;
    data.forEach((item) => {
        const index = item.lastIndexOf('.');
        if(item.substring(index) == '.txt') {
            setTimeout(downloadImage, counter, `${item.substring(0, index)}.jpg`, process.env.PROJECT_ID, 'exported/images');
            counter += parseInt(process.env.REQUEST_INTERVAL);
        }
    });
}
