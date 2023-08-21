import json
import os
import pandas as pd
import numpy as np
import time
import openai
import argparse

message = [{'role':'system', 'content': f"Suppose a person has performed the given actions in the form of a sequence of action pairs, there should be no more than two words in each action pair Each action pair is defined by a {{verb}} and a {{noun}}, separated by a space. What will be the possible next 20 actions? \\You should follow the following rules: 1.For each action pairs, you can only choose the {{verb}} from the following words: [adjust, apply, arrange, attach, blow, break, carry, catch, clap, clean, climb, close, consume, count, cover, crochet, cut, detach, dig, dip, divide, draw, drill, drive, enter, feed, file, fill, fold, fry, give, grate, grind, hang, hit, hold, insert, inspect, iron, kick, knead, knit, lift, lock, loosen, mark, measure, mix, mold, move, open, operate, pack, paint, park, peel, pet, plant, play, point, pour, press, pull, pump, push, put, read, remove, repair, roll, sand, scoop, scrape, screw, scroll, search, serve, sew, shake, sharpen, shuffle, sieve, sit, smooth, spray, sprinkle, squeeze, stand, step, stick, stretch, swing, take, talk, throw, tie, tighten, tilt, touch, turn, turn, turn, uncover, unfold, unroll, unscrew, untie, walk, wash, water, wear, weld, wipe, write, zip] \\ 2. For each action pairs, you can only choose the {{noun}} from the following words: [apple, apron, arm, artwork, asparagus, avocado, awl, axe, baby, bacon, bag, baking, ball, ball, balloon, banana, bar, baseboard, basket, bat, bat, bathtub, batter, battery, bead, beaker, bean, bed, belt, bench, berry, beverage, bicycle, blanket, blender, block, blower, bolt, book, bookcase, bottle, bowl, bracelet, brake, brake, branch, bread, brick, broccoli, broom, brush, bubble, bucket, buckle, burger, butter, butterfly, button, cabbage, cabinet, calculator, caliper, camera, can, candle, canvas, car, card, cardboard, carpet, carrot, cart, cat, ceiling, celery, cello, cement, cereal, chaff, chain, chair, chalk, cheese, chicken, chip, chip, chip, chisel, chocolate, chopping, chopstick, cigarette, circuit, clamp, clay, clip, clock, cloth, coaster, coconut, coffee, coffee, colander, comb, computer, container, cooker, cookie, cork, corn, corner, countertop, crab, cracker, crayon, cream, crochet, crowbar, cucumber, cup, curtain, cushion, cutter, decoration, derailleur, detergent, dice, dishwasher, dog, door, doorbell, dough, dough, doughnut, drawer, dress, drill, drill, drum, dumbbell, dust, duster, dustpan, egg, eggplant, engine, envelope, eraser, facemask, fan, faucet, fence, file, filler, filter, fish, fishing, flash, floor, flour, flower, foam, foil, food, foot, fork, fridge, fries, fuel, funnel, game, garbage, garlic, gasket, gate, gauge, gauze, gear, generator, ginger, glass, glasses, glove, glue, glue, golf, gourd, grain, grape, grapefruit, grass, grater, grill, grinder, guava, guitar, hair, hammer, hand, handle, hanger, hat, hay, haystack, head, headphones, heater, helmet, hinge, hole, horse, hose, house, ice, ice, ink, iron, jack, jacket, jug, kale, ketchup, kettle, key, keyboard, knife, label, ladder, leaf, leash, leg, lemon, lever, lid, light, lighter, lime, lock, lubricant, magnet, mango, manure, mask, mat, matchstick, meat, medicine, metal, microscope, microwave, milk, mirror, mixer, mold, money, mop, motorcycle, mouse, mouth, mower, multimeter, mushroom, nail, nail, nail, napkin, necklace, needle, net, nozzle, nut, nut, oil, okra, onion, oven, paddle, paint, paint, paintbrush, palette, pan, pancake, panel, pants, papaya, paper, pasta, paste, pastry, pea, peanut, pear, pedal, peel, peeler, peg, pen, pencil, pepper, phone, photo, piano, pickle, picture, pie, pillow, pilot, pin, pipe, pizza, planer, plant, plate, playing, plier, plug, pole, popcorn, pot, pot, potato, pump, pumpkin, purse, puzzle, rack, radio, rail, rake, razor, remote, rice, ring, rod, rolling, root, rope, router, rubber, ruler, sand, sander, sandpaper, sandwich, sauce, sausage, saw, scarf, scissors, scoop, scraper, screw, screwdriver, sculpture, seasoning, seed, set, sewing, sharpener, shears, sheet, shelf, shell, shirt, shoe, shovel, shower, sickle, sieve, sink, sketch, skirt, slab, snorkel, soap, sock, socket, sofa, soil, solder, soup, spacer, spatula, speaker, sphygmomanometer, spice, spinach, spirit, sponge, spoon, spray, spring, squeezer, stairs, stamp, stapler, steamer, steering, stick, sticker, stock, stone, stool, stove, strap, straw, string, stroller, switch, syringe, table, tablet, taco, tape, tape, tea, teapot, television, tent, test, tie, tile, timer, toaster, toilet, toilet, tomato, tongs, toolbox, toothbrush, toothpick, torch, towel, toy, tractor, trash, tray, treadmill, tree, trimmer, trowel, truck, tweezer, umbrella, undergarment, vacuum, vacuum, valve, vase, video, violin, wall, wallet, wallpaper, washing, watch, water, watermelon, weighing, welding, wheat, wheel, wheelbarrow, whisk, window, windshield, wiper, wire, wood, worm, wrapper, wrench, yam, yeast, yoghurt, zipper, zucchini].\\Remember the output must be exact 20 actions in the form of {{verb}} and a {{noun}}, which means there are 19 ',' in the output.\n"},]

demos = [
"scene: gardening ## observed actions: tie leaf, carry leaf, put plant, adjust cloth, take plant, put plant, take sickle, take sickle => cut plant, put sickle, take leaf, stretch rubber, take sickle, cut plant, hold plant, put sickle, take rubber, pull rubber, take rubber, tie rubber, move plant, take rubber, pull rubber, put plant, hold plant, cut plant, cut plant, hold plant ###\n",

"scene: making bricks ## observed actions: put cement, wipe mold, arrange mold, turn mold, put mold, take soil, pour mold, remove mold => put mold, wipe floor, cut cement, mix cement, arrange mold, put cement, remove cement, put cement, wipe cement, carry mold, put mold, turn mold, put mold, pour sand, put mold, cut clay, arrange mold, put clay, remove clay, carry mold ###\n",

"scene: baking bread ## observed actions: put dough, put dough, take dough, take dough, take dough, put dough, adjust dough, adjust tray => take dough, put dough, roll dough, take cutter, cut dough, put cutter, take dough, put dough, take dough, put dough, take dough, put dough, take dough, put dough, take dough, take dough, put dough, take brush, hold bowl, take bottle ###\n",

"scene: prune plant ## observed actions: take plier, hold plant, cut plant, cut plant, throw plant, cut plant, hold plant, cut plant => hold plant, cut plant, cut plant, cut plant, put plier, take plier, cut stick, put stick, put shears, take bucket, pour soil, put bucket, break soil, take vase, pour soil, put vase, take soil, take soil, throw soil, take soil ###\n",

"scene: chop carrots ## observed actions: take carrot, cut carrot, cut carrot, cut carrot, take carrot, put carrot, cut carrot, cut carrot => cut carrot, take carrot, put carrot, cut carrot, cut carrot, put carrot, move carrot, take carrot, cut carrot, cut carrot, cut carrot, put carrot, cut carrot, cut carrot, cut carrot, take carrot, put carrot, put carrot, take carrot, cut carrot ###\n",

"scene: painting metal ## observed actions: put paper, put metal, take paper, adjust paper, put paper, adjust paper, take paintbrush, dip paintbrush => apply paper, dip container, apply paper, dip container, apply paper, dip container, apply paper, take paper, put paper, take paper, stick paper, smooth paper, take paper, put paper, fold paper, turn paper, smooth paper, turn paper, smooth paper, turn paper ###\n",

"scene: make drink ## observed actions: take lid, close container, put container, close drawer, mix beverage, take bottle, put bottle, pet dog => hold cup, open bottle, take bottle, close bottle, put bottle, mix beverage, push bottle, take cup, mix beverage, consume beverage, put cup, take bottle, put bottle, take cabinet, pour whisk, close bottle, put whisk, take bottle, open fridge, put bottle ###\n",

"scene: woodworking ## observed actions: tighten clamp, sand sander, sand wood, sand wood, wipe wood, sand wood, sand wood, sand wood => wipe wood, wipe wood, sand wood, wipe wood, sand wood, sand wood, sand wood, sand wood, sand wood, sand wood, sand wood, sand wood, sand wood, sand wood, sand wood, sand wood, wipe wood, wipe wood, sand wood, sand wood ###\n",

"scene: clean dishes ## observed actions: take sponge, take soap, put sponge, put soap, take plate, wash plate, wash plate, put plate => close faucet, take chopstick, adjust bag, adjust zucchini, hold pan, take napkin, clean countertop, put napkin, adjust zucchini, adjust zucchini, take napkin, put chopstick, open faucet, wash napkin, squeeze napkin, close faucet, clean napkin, put napkin, turn zucchini, take lid ###\n",

"scene: sew clothes ## observed actions: take scissors, move cloth, put cloth, take cloth, put cloth, hold cloth, cut cloth, give scissors => give scissors, put scissors, lift cloth, put cloth, fold cloth, take cloth, attach cloth, put cloth, move cloth, adjust sewing-machine, adjust cloth, turn wheel, sew cloth, move cloth, take scissors, give scissors, give scissors, put scissors, adjust cloth, sew cloth ###\n",

"scene: make bread ## observed actions: take dough, mold dough, put dough, take dough, mold dough, put dough, take dough, mold dough => mold dough, put dough, take dough, mold dough, put dough, take dough, cut dough, put dough, put dough, put dough, put dough, take dough, cut dough, mold dough, put pan, adjust dough, take dough, cut dough, mold dough, put dough ###\n",

"scene: fix wheel ## observed actions: put nut, put drill, take nut, screw nut, hold wheel, screw screw, hold screw, take drill => insert screw, hold drill, put drill, take wrench, remove nut, insert nut, tighten nut, put bolt, take screwdriver, take nut, put screwdriver, insert screw, screw screw, insert screw, screw screw, screw screw, screw screw, shake wheel, put screwdriver, take wrench ###\n",
    
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--response_dir', type=str, default="output", help="path to the output directory")
    parser.add_argument('--dataset_dir', type=str, default="dataset", help="path to the dataset directory")
    parser.add_argument('--train_name', type=str, default="train_nseg8.jsonl", help="name of the training file")
    parser.add_argument('--val_name', type=str, default="val_nseg8_recog_subset600.jsonl", help="name of the validation file")
    parser.add_argument('--response_name', type=str, default="subset600_responses_icl_recog_goal.json", help="name of the output file")
    parser.add_argument('--goal_name', type=str, default="goal_val600.json", help="name of the goal file")
    parser.add_argument('--openai_key', type=str, required=True, help="openai key")
    parser.add_argument('--gpt_model', type=str, default='gpt-3.5-turbo', help="GPT api name")
    parser.add_argument('--temperature', type=float, default=0.3, help="GPT api parameter temperature")
    parser.add_argument('--n', type=int, default=5, help="GPT api parameter n")
    parser.add_argument('--max_tokens', type=int, default=500, help="GPT api parameter max_tokens")
    args = parser.parse_args()

    response_path = os.path.join(args.response_dir, args.response_name)
    train_path = os.path.join(args.dataset_dir, args.train_name)
    val_path = os.path.join(args.dataset_dir, args.val_name)
    goal_path = os.path.join(args.dataset_dir, args.goal_name)
    
    print('training data path: ', train_path)
    print('validation data path: ', val_path)
    print('goal file path: ', goal_path)
    print('Response saving path: ', response_path)

    train_df = pd.read_json(train_path, lines=True)
    val_df = pd.read_json(val_path, lines=True)
    goal_list = json.load(open(goal_path,"r"))

    train_x = train_df['prompt'].apply(lambda x: x.replace("\n","").replace("#","")[:-1]).tolist()
    train_y = train_df['completion'].apply(lambda x: x.strip().replace("\n","").replace("#","")[:-1]).tolist()
    val_x = val_df['prompt'].apply(lambda x: x.replace("\n","").replace("#","")[:-1]).tolist()
    val_y = val_df['completion'].apply(lambda x: x.strip().replace("\n","").replace("#","")[:-1]).tolist()

    openai.api_key = args.openai_key
    print('OpenAI key ending with: ', openai.api_key[-5:])

    val_idx = np.arange(len(val_x)).tolist()
    total_num = len(val_idx)
    over = False
    print("start ICL querying from {}".format(args.gpt_model))
    print('GPT api parameters: ', args.temperature, args.n, args.max_tokens)
    while not over:
        try:
            answers_list = []
            answer_len_list = []
            goals_list = []
            try:
                responses_list = json.load(open(response_path, "r"))
            except:
                responses_list = []
                json.dump(responses_list, open(response_path, "w"))
            print("processed sample num: ", len(responses_list)) 

            for ii, prompt_idx in enumerate(val_idx):
                if ii < len(responses_list):
                    continue
                prompt_message = ""
                for demo in demos:
                    prompt_message += demo
                prompt_message += "scene: " + goal_list[prompt_idx] + " ## observed actions: "
                prompt_message += val_x[prompt_idx] + " => "
                mes = message + [{'role':'user', 'content': prompt_message}]
                response = openai.ChatCompletion.create(
                                                model=args.gpt_model,
                                                messages=mes,
                                                max_tokens = args.max_tokens,
                                                n = args.n,
                                                temperature = args.temperature,
                                                )
                choices = response['choices']
                answers = []
                answer_len = []
                goals = []
                res_list = []
                for choice, i in zip(choices, range(len(choices))):
                    res_list.append(choice["message"]["content"])
                    try:
                        answer = choice["message"]["content"].strip().split(", ")
                        answer_len.append(len(answer))
                    except:
                        print('fail to parse')
                print(str(ii+1)+'/'+str(total_num), answer_len)
                responses_list.append(res_list)
                json.dump(responses_list, open(response_path, "w"))
            over = True
        except Exception as e:
            print(e)
            print("Rate limit error, sleep for 30 seconds")
            time.sleep(30)