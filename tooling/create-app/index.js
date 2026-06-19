import fs from "node:fs";
import path from "node:path";

const [, , type, name] = process.argv;

if (!type || !name) {
  console.error("Usage: create-app <type> <name>");
  process.exit(1);
}

const templateDir = path.resolve(`tooling/create-app/templates/${type}`);
const targetDir = path.resolve(`apps/${name}`);

fs.cpSync(templateDir, targetDir, { recursive: true });

const files = fs.readdirSync(targetDir, {
  recursive: true,
  withFileTypes: true,
});
files
  .filter((f) => f.isFile())
  .map((f) => `${f.parentPath}/${f.name}`)
  .forEach((f) => {
    const data = fs.readFileSync(f, "utf-8");
    const content = data.replace(/__APP_NAME__/g, name);
    fs.writeFileSync(f, content);
  });

console.log(`Created ${type} app: ${name}`);
