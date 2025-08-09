import fs from "fs";
import path from "path";
import { RAG } from "./rag.js";
import chalk from "chalk";

async function main() {
  const dataPath = path.join(process.cwd(), "data.json");
  // Huge data sample -> but we get irrelevent response bcoz data set is not correct
  // const dataPath = path.join(process.cwd(), "data2.json");
  const data = JSON.parse(fs.readFileSync(dataPath, "utf-8"));

  const rag = new RAG(data);
  await rag.init();

  const userQuery = process.argv.slice(2).join(" ");
  if (!userQuery) {
    console.log("Please provide a query as argument.");
    process.exit(1);
  }

  const answer = await rag.answerQuestion(userQuery, 1);
  console.log(chalk.yellowBright("\n=== Answer ==="));
  console.log(chalk.greenBright(answer));
}

main();
